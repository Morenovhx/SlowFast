from slowfast.utils.misc import get_class_names
from slowfast.utils.parser import load_config, parse_args
import numpy as np
import json
import csv
args = parse_args()

cfg = load_config(args)
class_names_path = cfg.DEMO.LABEL_FILE_PATH #ava_classnames.json
mode = cfg.DEMO.VIS_MODE #top-k or thres
common_class_thres = cfg.DEMO.COMMON_CLASS_THRES #0.7
uncommon_class_thres = cfg.DEMO.UNCOMMON_CLASS_THRES #0.3
common_class_names = cfg.DEMO.COMMON_CLASS_NAMES
top_k = cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS #1
predictions_path = cfg.DEMO.PREDICTIONS_FILE_PATH
scores_path = cfg.DEMO.SCORES_FILE_PATH

def get_best_predictions(scores):
    """
        Choose which predictions to keep based on the config file
        Args:
            scores (dict): dict contains the score for each class
    """
    scores = {class_name:round(score,4) for class_name, score in scores.items()}

    if mode == 'top-k':
        if top_k>len(scores):
            return scores
        else:
            return {class_name:score for (class_name, score) in list(scores.items())[0:top_k]}
    elif mode == 'thres':
        temp = scores.copy()
        for class_name, score in temp.items():
            if score < common_class_thres and class_name in common_class_names:
                scores.pop(class_name)
            elif score < uncommon_class_thres and class_name not in common_class_names:
                scores.pop(class_name)
        return scores

class_names, _, _ = get_class_names(class_names_path)
file = np.genfromtxt(scores_path,delimiter=',')

task_predictions = {}
person_predictions = {}
total_predictions = {}
total_predictions['total'] = 0
frame_number = file[0][0]
actions = []
for line in file:
    # get prediction scores
    prediction_scores = {class_name:score for class_name, score in zip(class_names, line[7:])}
    prediction_scores = dict( sorted(prediction_scores.items(),
                           key=lambda item: item[1],
                           reverse=True))
    prediction_scores = get_best_predictions(prediction_scores)                           
    timestamp = int(round(((line[0]-128)+line[1])/25))+46
    box_coords = {'bot_left_x':line[3],
                  'bot_left_y':line[4],
                  'top_right_x':line[5],
                  'top_right_y':line[6]}
                  
    action = {'timestamp': timestamp,
                'bot_left_x': line[3],
                'bot_left_y': line[4],
                'top_right_x': line[5],
                'top_right_y': line[6],
                'tags': list(prediction_scores.keys())}
    actions.append(action)
    # assign predictions by person
    box_id = int(line[2])
    if box_id not in list(person_predictions.keys()): 
        person_predictions[box_id] = {'total':0}
    
    person_predictions[box_id]['total'] += 1

    if not prediction_scores.keys():
        if 'nothing' in person_predictions[box_id].keys():   
            person_predictions[box_id]['nothing'] += 1
        else:
            person_predictions[box_id]['nothing'] = 1

    for prediction in prediction_scores.keys():
        if prediction in person_predictions[box_id].keys():
            person_predictions[box_id][prediction] += 1
        else:
            person_predictions[box_id][prediction] = 1
    # assign predictions total
    total_predictions['total'] += 1

    if not prediction_scores.keys():
        if 'nothing' in total_predictions.keys():   
            total_predictions['nothing'] += 1
        else:
            total_predictions['nothing'] = 1

    for prediction in prediction_scores.keys():
        if prediction in total_predictions.keys():
            total_predictions[prediction] += 1
        else:
            total_predictions[prediction] = 1   
    # assign predictions by task 
    task_id = int(line[0]/frame_number)
    if task_id not in task_predictions.keys():
        task_predictions[task_id] = {}

    task_predictions[task_id][box_id] = prediction_scores

predictions = {'total':total_predictions, 'by person': person_predictions, 'by task':task_predictions}
with open("new_scores.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile,delimiter=',')
    for action in actions:
        for tag in action['tags']:
            csvwriter.writerow([action['timestamp'],action['bot_left_x'],action["bot_left_y"],action["top_right_x"],action["top_right_y"],tag])
with open(predictions_path, 'w') as f:
    json.dump(predictions, f, indent=4)