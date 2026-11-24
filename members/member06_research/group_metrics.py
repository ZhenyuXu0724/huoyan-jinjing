"""Image-level evaluation groups from annotation metadata, never predicted object counts."""
from collections import Counter, defaultdict


def decision_metrics(rows, *, allow_dev_only=False):
    matrix=[[0,0],[0,0]];support=Counter();states=Counter();decided=0
    for row in rows:
        if row['label'] not in {'PASS','FAIL'}:raise ValueError('invalid ground truth label')
        support[row['label']]+=1;states[row['status']]+=1
        if row['status']=='OK' or (allow_dev_only and row['status']=='DEV_ONLY') or row['status']=='SMOKE_ONLY':
            if row['predicted_label'] not in {'PASS','FAIL'}:raise ValueError('valid decision requires PASS/FAIL')
            matrix[row['label']=='FAIL'][row['predicted_label']=='FAIL']+=1;decided+=1
    tn,fp=matrix[0];fn,tp=matrix[1];total=len(rows)
    return {'n':total,'decided':decided,'coverage':decided/total if total else None,'status_counts':dict(states),
        'support':{k:support[k] for k in ('PASS','FAIL')},'confusion_matrix':matrix,'matrix_order':['PASS','FAIL'],
        'false_pass':fn,'false_fail':fp,'fail_precision':tp/(tp+fp) if tp+fp else None,
        'fail_recall':tp/(tp+fn) if tp+fn else None,'fail_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        'fail_recall_all_requests':tp/support['FAIL'] if support['FAIL'] else None,
        'accuracy_decided':(tp+tn)/decided if decided else None,
        'accuracy_all_requests':(tp+tn)/total if total else None}


def grouped_validation(rows, predictions, config):
    if len(rows)!=len(predictions):raise ValueError('row/prediction identity mismatch')
    labels={v:k for k,v in config.class_to_idx.items()};cases=[]
    status=config.evidence_status if config.evidence_status in {'DEV_ONLY','SMOKE_ONLY'} else 'OK'
    for row,pred in zip(rows,predictions):
        if labels[pred['target_index']]!=row['label']:raise ValueError('prediction target differs from dataset')
        cases.append({'image_path':str(row['path']),'scene_id':row['scene_id'],'label':row['label'],
            'predicted_label':labels[pred['predicted_index']],'status':status,
            'fail_softmax':pred['softmax'][config.class_to_idx['FAIL']],
            'anomaly_type':row.get('anomaly_type','UNAVAILABLE'),
            'non_catalogue_count':row.get('non_catalogue_count','UNAVAILABLE'),
            'source_package':row.get('source_package','UNAVAILABLE'),
            'capture_condition':row.get('capture_condition','UNAVAILABLE')})
    grouped={}
    for field in ('anomaly_type','non_catalogue_count','source_package','capture_condition'):
        groups=defaultdict(list)
        for row in cases:groups[str(row[field])].append(row)
        grouped[field]={key:decision_metrics(values,allow_dev_only=config.training_tier=='DEV_ONLY') for key,values in groups.items()}
    return {'evidence_status':config.evidence_status,'metadata_basis':'annotation-derived groups, not model detections or actual count predictions',
            'overall':decision_metrics(cases,allow_dev_only=config.training_tier=='DEV_ONLY'),'groups':grouped,
            'errors':[r for r in cases if r['label']!=r['predicted_label']], 'cases':cases}
