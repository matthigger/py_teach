import copy
import json
import pathlib

ACTION_ERROR = 0
ACTION_RM_CELL = 1
ACTION_CLEAR_CELL = 2

empty_code_cell = {'cell_type': 'code',
                   'execution_count': None,
                   'metadata': {},
                   'outputs': [],
                   'source': []}
empty_markdown_cell = {'cell_type': 'markdown',
                       'metadata': {},
                       'source': []}


# todo: need action class (negate, target, event)
# todo: CLI this whole thing

def json_from_ipynb(file):
    file = pathlib.Path(file)

    with open(str(file), 'r') as f:
        return json.load(f)


def search_cell(json_dict, target):
    # we search back to front so cell_idx is valid after deletion
    for cell_idx, cell in reversed(list(enumerate(json_dict['cells']))):
        line_list = list()
        for line in cell['source']:
            if target in line.lower():
                line_list.append(line)
        if line_list:
                yield cell_idx, line_list


def search_act(json_dict, target_act_list):
    """ returns copy of json_dict. actions taken on all cells with target

    Args:
        json_dict (dict): dictionary of ipynb file
        target_act_list (list): list of tuples.  each tuple is a target string
            and an action (see ACTION_RM_CELL, ACTION_ERROR)

    Returns:
        json_dict (dict): dictionary of ipynb file, actions taken on cells
            where target found
    """
    # leave original intact
    json_dict = copy.copy(json_dict)

    for target, action in target_act_list:
        for cell_idx, line_list in search_cell(json_dict, target):
            if action == ACTION_ERROR:
                s_error = f'{target} found in cell {cell_idx}: {line_list}'
                raise AttributeError(s_error)
            elif action == ACTION_RM_CELL:
                # delete cell
                del json_dict['cells'][cell_idx]
            elif action == ACTION_CLEAR_CELL:
                # replaces cell with an empty cell (of same type)
                cell_type = json_dict['cells'][cell_idx]['cell_type']
                if cell_type == 'code':
                    json_dict['cells'][cell_idx] = empty_code_cell
                elif cell_type == 'markdown':
                    json_dict['cells'][cell_idx] = empty_markdown_cell
                else:
                    raise AttributeError(
                        f'unrecognized cell type: {cell_type}')
            else:
                raise AttributeError(f'action not recognized: {action}')

    return json_dict


def quiz_hw_prep(file, stem='_rub.ipynb'):
    new_file_dict = {'_sol.ipynb': [('rubric', ACTION_RM_CELL),
                                    ('todo', ACTION_ERROR)],
                     '.ipynb': [('rubric', ACTION_RM_CELL),
                                ('solution', ACTION_RM_CELL),
                                ('todo', ACTION_ERROR)]}

    if stem not in str(file):
        raise IOError('file must be named f{stem}')

    return prep(file, new_file_dict, stem=stem)


def notes_prep(file, stem='.ipynb'):
    new_file_dict = {'_stud.ipynb': [('solution', ACTION_CLEAR_CELL),
                                     ('todo', ACTION_ERROR)]}
    # todo: ICA version too
    return prep(file, new_file_dict, stem=stem)


def prep(file, new_file_dict, stem):
    json_dict = json_from_ipynb(file)
    for new_file, target_act_list in new_file_dict.items():
        _json_dict = search_act(json_dict, target_act_list)

        new_file = pathlib.Path(str(file).replace(stem, new_file))
        print(f'building: {new_file}')

        with open(str(new_file), 'w') as f:
            json.dump(_json_dict, f)


if __name__ == '__main__':
    # import sys
    # file = pathlib.Path(sys.argv[1]).resolve()
    file = '/home/matt/Dropbox/teach/DS3000/quiz/quiz3/quiz3_rub.ipynb'
    quiz_hw_prep(file)
