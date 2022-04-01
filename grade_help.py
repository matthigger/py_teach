import pathlib
from collections import defaultdict
from warnings import warn

import numpy as np
import pandas as pd


def get_df_assign(df_scope, cat_weight_dict, exclude_list=None):
    # groom cat_weight_dict (lowername categories & normalized weights)
    total_weight = sum(cat_weight_dict.values())
    cat_weight_dict = {cat.lower(): weight / total_weight
                       for cat, weight in cat_weight_dict.items()}

    # index into only columns with 'Max Points' (one per assignmnet)
    columns = tuple(col for col in df_scope.columns if 'Max Points' in col)
    df_assign = df_scope.loc[:, columns].drop_duplicates()
    assert df_assign.shape[0] == 1, \
        'Max Points has multiple values for a given assignment'

    # process assignment names
    def get_assignment_name(s):
        return s.replace(' - Max Points', '').strip().lower()

    df_assign = df_assign.rename(axis=1, mapper=get_assignment_name)

    # groom df_assign
    df_assign = df_assign.transpose().reset_index()
    df_assign.columns = ['name', 'max pts']

    # identify which assignments included
    exclude_list = [s.lower() for s in exclude_list]

    def is_included(name):
        for s in exclude_list:
            if s in name:
                return False
        return True

    df_assign['include'] = df_assign['name'].apply(is_included)

    # get category per assignment
    def get_category(name):
        cat = None
        for _cat in cat_weight_dict.keys():
            if _cat in name:
                assert cat is None, \
                    f'2 categories for assignment {name}: ({cat} & {_cat})'
                cat = _cat
        return cat

    df_assign['category'] = df_assign['name'].apply(get_category)

    # compute pts per category
    df_assign['max pts included'] = \
        df_assign['max pts'].multiply(df_assign['include'])
    pts_per_cat = df_assign.groupby('category').sum()['max pts included']
    del df_assign['max pts included']

    # get weight of each assignment
    def normalize_assign(row):
        cat = row['category']
        if not row['include']:
            # explicilty excluded assignment
            return 0
        if cat not in pts_per_cat:
            name = row['name']
            warn(f'implicitly excluded assignment, no category match: {name}')
            return 0
        return row['max pts'] / pts_per_cat[cat] * cat_weight_dict[cat]

    df_assign['weight'] = df_assign.apply(normalize_assign, axis=1)
    assert np.isclose(df_assign['weight'].sum(), 1), 'weights not normalized'

    # get mean score (percent) per assignment
    df_scope = df_scope.rename(axis=1, mapper=str.lower)
    for idx, assign in df_assign.iterrows():
        name = assign['name']
        max_pts = assign['max pts']
        if pd.isnull(df_scope.loc[:, name]).all() and assign['include']:
            warn(f'assignment grades missing (counting as 0): {name}')
        df_assign.loc[idx, 'mean (completed)'] = df_scope.loc[:, name].mean(skipna=True) / max_pts
        df_assign.loc[idx, 'complete'] = 1 - pd.isna(df_scope.loc[:, name]).sum() / df_scope.shape[0]

    df_assign.sort_values(['weight', 'name'], ascending=False, inplace=True)

    return df_assign


def get_df_grade(df_scope, df_assign, cat_drop_n=None, fallback_assign=None,
                 f_waive=None):
    if cat_drop_n is None:
        cat_drop_n = defaultdict(lambda: 0)

    # only examine included assignments
    df_assign = df_assign.loc[df_assign['include'], :]

    # lowercase assignment names (and make copy)
    df_scope = df_scope.rename(axis=1, mapper=str.lower)
    df_scope.set_index('email', inplace=True)
    # all missing or waived assignments are zeros (waived have nan max points)
    df_scope.fillna(0, inplace=True)

    # init df_grade from student meta data in df_scope
    col_keep = ('last name', 'first name', 'sid', 'section_name')
    df_grade = df_scope.loc[:, col_keep]

    # to_max_pts
    assign_max_pts_list = [col for col in df_scope.columns
                           if ' - max points' in col]

    def to_max_pts(s_assign):
        """ gets column of max pts per assignment """
        max_pts_all = [col for col in assign_max_pts_list
                       if s_assign.lower().strip() in col]
        if len(max_pts_all) != 1:
            s_error = f'no unique assignment: {s_assign} in {max_pts_all}'
            raise RuntimeError(s_error)

        return max_pts_all[0]

    # waive assignments (set max score to nan)
    if f_waive is not None:
        df_waive = pd.read_csv(str(f_waive), index_col='email')
        for idx, row in df_waive.iterrows():
            for assign in row['assign'].split(','):
                df_scope.loc[idx, to_max_pts(assign)] = np.nan

    # incorporate fallback assignments (used in place of another assignment
    # when score is higher)
    if fallback_assign is not None:
        for name_to, name_from in fallback_assign.items():
            _df = df_scope.loc[:, (name_to, name_from)]
            df_scope[name_to] = _df.max(axis=1)

    # compute percentage on each assignment
    for _, assign in df_assign.iterrows():
        # student average on assignment
        name = assign['name']
        df_grade[name] = df_scope.loc[:, name] / assign['max pts']

        # todo: late penalties

    # average per category (dropping lowest)
    s_cat_namelist = df_assign.groupby('category')['name'].apply(list)
    _df_assign = df_assign.set_index('name')
    for cat, assign_list in s_cat_namelist.iteritems():
        for idx, student in df_grade.iterrows():
            perc = student[assign_list].to_numpy().astype(float)
            assign_max_list = list(map(to_max_pts, assign_list))
            weight = df_scope.loc[idx, assign_max_list].to_numpy().astype(
                float)

            df_grade.loc[idx, 'mean_' + cat] = \
                get_mean_drop_low(perc, weight, drop_n=cat_drop_n[cat])

    # overall average
    df_grade['mean'] = 0
    s_cat_weight = df_assign.groupby('category').sum()['weight']
    for cat, weight in s_cat_weight.iteritems():
        df_grade['mean'] += df_grade['mean_' + cat] * weight

    # get missing assignments
    assign_list = list(
        df_assign.sort_values('weight', ascending=False)['name'])

    def get_missing(row):
        return tuple(name for name in assign_list if not row[name])

    df_grade['missing'] = df_grade.apply(get_missing, axis=1)

    # assign letter grade
    df_grade['letter'] = df_grade['mean'].apply(val_to_letter)

    # shuffle column order to bring most relevant info to earlier columns
    cat_list = s_cat_weight.index
    col_list = list(col_keep) + ['mean', 'letter'] + ['mean_' + cat
                                                      for cat in
                                                      s_cat_weight.index]

    col_list += sorted(set(df_grade.columns) - set(col_list))
    df_grade = df_grade.loc[:, col_list]

    return df_grade


def get_mean_drop_low(perc, weight, drop_n=0):
    """ drops lowest assignment, returns mean

    note: this doesn't necessarily maximize grade given weight ... might be
    worth optimizing down the road but its not obvious (to me) how to do this
    and the difference is minimal (is documented as lowest grade to students)

    Args:
        perc (np.array): percentage earned per assignment
        weight (np.array): weight of each assignment
        drop_n (int): number of assignments to drop

    Returns:
        mean (float): mean score, weighted by weight after having dropped the
            most damaging drop_n assignments
    """
    weight_init = weight
    perc_init = perc

    # drop nans
    idx_keep = np.logical_and(~np.isnan(weight),
                              ~np.isnan(perc))
    weight = weight[idx_keep]
    perc = perc[idx_keep]

    # drop a few assignments
    idx_keep = np.argsort(perc)[drop_n:]
    weight = weight[idx_keep]
    perc = perc[idx_keep]

    if not weight.size:
        # no assignments to average
        return np.nan

    # compute weighted average
    return np.inner(perc, weight) / weight.sum()


def val_to_letter(val, grade_thresh=None):
    if grade_thresh is None:
        grade_thresh = {'A': .93,
                        'A-': .90,
                        'B+': .87,
                        'B ': .83,
                        'B-': .80,
                        'C+': .77,
                        'C ': .73,
                        'C-': .70,
                        'D+': .67,
                        'D ': .63,
                        'D-': .60,
                        'E ': .0}

    if np.isnan(val):
        return 'no-grade'

    for mark, thresh in grade_thresh.items():
        if val >= thresh:
            return mark

    raise IOError('invalid input')


if __name__ == '__main__':
    # todo: move to pytest and test all other fnc
    mean = get_mean_drop_low(perc=np.array([1, 1, 1]),
                             weight=np.array([1, 1, 1]), drop_n=0)
    mean_expect = 1

    assert mean == mean_expect

    mean = get_mean_drop_low(perc=np.array([1, .89, .9]),
                             weight=np.array([1, 1, 10]), drop_n=1)
    mean_expect = 10 / 11

    assert mean == mean_expect

    mean = get_mean_drop_low(perc=np.array([1, .89, .9]),
                             weight=np.array([1, np.nan, 10]), drop_n=1)
    mean_expect = 1

    assert mean == mean_expect
