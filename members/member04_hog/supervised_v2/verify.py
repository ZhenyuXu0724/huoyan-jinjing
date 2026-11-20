"""Real-weight verification, complete experiment evidence, and immutable v1 audit."""
import json
import tempfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ..model import load_model,predict_with_model
from ..data import sha256
from .common import ROOT,DATA,OUT,WEIGHTS,GROUPS,SEEDS,read,write,check_old
from .inference import batch,predict
from .assess import verify_freeze,csv_rows


def interface_smoke(image):
    with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=1):
        root=Path(temp);good=root/'good.jpg';good.write_bytes(image.read_bytes())
        bad=root/'bad.png';bad.write_text('invalid image')
        result=predict(good)
        assert result['status']=='OK' and result['score_type']=='uncalibrated_fail_probability'
        assert predict(bad)['status']=='INPUT_ERROR'
        assert predict(root/'missing.jpg')['status']=='INPUT_ERROR'
        assert predict(good,root/'absent.joblib')['status']=='MODEL_UNAVAILABLE'
        counts={str(i):2 for i in range(1,6)}
        cases=[dict(image_id='good.jpg',expected_counts=counts),dict(image_id='bad.png',expected_counts=counts),
               dict(image_id='missing.jpg',expected_counts=counts),
               dict(image_id='unsupported.jpg',expected_counts={**counts,'1':3})]
        request=root/'counts.json';request.write_text(json.dumps(cases))
        rows=batch(root,request,root/'out')
        assert [r['status'] for r in rows]==['OK','INPUT_ERROR','INPUT_ERROR','UNSUPPORTED_COUNTS']
        assert rows[0]['predicted_label']==result['predicted_label']
        assert all(r['predicted_label'] is None and r['prediction_score'] is None for r in rows[1:])
        config=read(root/'out/run_config.json')
        assert config['score_type']=='uncalibrated_fail_probability'
        assert config['model_version']=='baseline1_supervised_v2'
        try:batch(root,request,root/'missing_model',root/'absent.joblib')
        except FileNotFoundError:assert not (root/'missing_model').exists()
        else:raise AssertionError('Missing model must fail')
        request.write_text(json.dumps([dict(cases[0],label='PASS')]))
        try:batch(root,request,root/'leak')
        except ValueError:pass
        else:raise AssertionError('Truth must be rejected')
    return dict(real_input='PASS',bad_missing_image='PASS',missing_model='PASS',
                unsupported_counts='PASS',truth_rejection='PASS',batch_export='PASS')


def run():
    frozen=verify_freeze()
    runs=0
    for group in GROUPS:
        for seed in SEEDS:
            directory=OUT/'experiments'/group/f'seed{seed}'
            artifact=WEIGHTS/'experiments'/group/f'seed{seed}/model.joblib'
            meta=read(directory/'metrics.json');history=read(directory/'history.json')
            assert sha256(artifact)==meta['model_sha256']
            assert len(history)==200 and all(np.isfinite(r['train_loss']) for r in history)
            assert len(csv_rows(directory/'valid_predictions.csv'))==101
            config=read(directory/'config.json')
            assert all(config[k]==v for k,v in GROUPS[group].items())
            assert abs(config['class_weights']['0']-185/266)<1e-12
            assert abs(config['class_weights']['1']-185/104)<1e-12
            runs+=1
    for left,right in [('B2','C_without_geometry'),('A2','C_without_clahe')]:
        for seed in SEEDS:
            a=csv_rows(OUT/'experiments'/left/f'seed{seed}/valid_predictions.csv')
            b=csv_rows(OUT/'experiments'/right/f'seed{seed}/valid_predictions.csv')
            assert a==b, 'Equivalent settings must reproduce identical predictions'
    sources=read(DATA/'provenance.json')['sources']
    valid={Path(r['image_path']).name:DATA/r['image_path'] for r in sources if r['split']=='valid'}
    bundle=load_model(ROOT/frozen['model_path'])
    saved=csv_rows(OUT/'experiments'/frozen['selected_group']/'seed42/valid_predictions.csv')
    differences=[]
    with threadpool_limits(limits=1):
        for row in saved:
            prediction=predict_with_model(valid[row['image_id']],bundle)
            assert prediction['status']=='OK'
            assert prediction['predicted_label']==row['predicted_label']
            difference=abs(prediction['prediction_score']-float(row['prediction_score']))
            assert np.isclose(prediction['prediction_score'],float(row['prediction_score']),rtol=1e-5,atol=1e-6)
            differences.append(difference)
    smoke=interface_smoke(next(iter(valid.values())))
    check_old()
    write(OUT/'verification.json',dict(experiments=runs,epochs_each=200,reloaded_validation_images=101,
          label_mismatches=0,max_score_difference=max(differences),duplicate_setting_predictions='IDENTICAL',
          old_preservation='PASS',interface=smoke))
    print('33 experiments, 101 reload predictions, real interface and v1 preservation verified',flush=True)


if __name__=='__main__':run()
