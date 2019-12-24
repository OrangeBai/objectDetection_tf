def gen_new_label(label_store):
    cls = {'backgroud': 0}
    new_label_store = []
    for cur_data in label_store:
        new_label = {'name': cur_data['name'], 'labels': []}
        for label in cur_data['labels']:
            if 'box2d' in label.keys():
                cur_label = {}
                category = label['category']
                if category not in cls.keys():
                    cls[category] = len(cls)
                cur_label['category'] = cls[category]
                cur_label['coordinates'] = (label['box2d']['x1'], label['box2d']['y1'],
                                            label['box2d']['x2'], label['box2d']['y2'])
                new_label['labels'].append(cur_label)
        new_label_store.append(new_label)
    return new_label_store, cls
