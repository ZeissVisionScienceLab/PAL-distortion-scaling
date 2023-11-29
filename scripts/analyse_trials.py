from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import cblearn as cbl
import cblearn.utils
import cblearn.embedding


MEASUREMENT_DIR = Path(__file__).parent / '../measurements/'
DATA_DIR = Path(__file__).parent / '../data/'


def get_data_file(subj):
    # check for all existing output files (from multiple sessions)
    # take the last session (should contain also answers from previous session)
    max_rep = -1
    for file in os.listdir(MEASUREMENT_DIR / str(subj).zfill(3)):
        if 'tracking' in file: # skip all tracking files
            continue
        if file.startswith(str(subj).zfill(3)):
            if file == str(subj).zfill(3) + '.csv':
                rep = 0
            else:
                rep = int(file[-6:-4])
            if rep > max_rep:
                max_rep = rep
                file_name = file
    return Path(MEASUREMENT_DIR, str(subj).zfill(3), file_name)


def read_measurements(path):
    # TODO check for all existing measurement files (if we plan to have one file per session)
    return pd.read_csv(path, delimiter=';')


def get_trial_df(subj):
    data_file = get_data_file(subj)
    data = read_measurements(data_file)
    # remove all trials with answer == 0 (not answered yet)
    data = data.drop(data[data.answer == 0].index)
    return data


def preprocess_queries(data):
    index_trivial_trials = (data.A_sph == 8) | (data.B_sph == 8) | (data.C_sph == 8)
    data = data[~index_trivial_trials] # trivial trials removed

    (triplets, responses), (query_trans, response_trans) = cbl.preprocessing.query_from_columns(
        data, [["A_sph", "A_add"], ["B_sph", "B_add"], ["C_sph", "C_add"]], 'answer', response_map={1: 1, 2: -1},
        return_transformer=True)
    objects = pd.DataFrame(query_trans.encoder_.classes_, columns=['sph', 'add'])
    queries = cbl.utils.check_query_response(triplets, responses, 'list-order')
    return objects, queries


def get_performance(data):
    # find trivial trials and check which are correct
    index_trivial_trials = (data.A_sph == 8) | (data.B_sph == 8) | (data.C_sph == 8)
    trivial_trials = data[index_trivial_trials].copy()  # contains only trivial trials
    trivial_trials["correct"] = (np.abs(trivial_trials.A_sph - trivial_trials.B_sph) < np.abs(trivial_trials.A_sph - trivial_trials.C_sph)) != (trivial_trials.answer == 2)
    trivial_perform = trivial_trials.correct.sum() / trivial_trials.shape[0]
    #TODO consistency flipped trials
    return trivial_perform

def get_flipped_consistency(data):
    # remove trivial trials
    index_trivial_trials = (data.A_sph == 8) | (data.B_sph == 8) | (data.C_sph == 8)
    data = data[~index_trivial_trials]  # trivial trials removed
    (triplets, responses), (query_trans, response_trans) = cbl.preprocessing.query_from_columns(
        data, [["A_sph", "A_add"], ["B_sph", "B_add"], ["C_sph", "C_add"]], 'answer', response_map={1: 1, 2: -1},
        return_transformer=True)
    any_flipped = np.full(len(triplets), False)
    no_flipped_found = 0
    for ix in range(len(triplets)):
        if any_flipped[ix]:
            continue

        flipped_ix = np.nonzero((triplets[ix, [0, 2, 1]] == triplets).all(axis=1))[0]
        if len(flipped_ix) != 1:
            no_flipped_found += 1
        else:
            any_flipped[flipped_ix] = True
            data.loc[ix, 'flipped_row'] = flipped_ix[0]

    print(f"Missed {no_flipped_found}/{len(triplets)} flipped triplets ACB.")
    orig_data = data[~data.flipped_row.isnull()]  # only get rows, where we have a flipped
    inv_responses = (orig_data.answer.values != data.iloc[orig_data.flipped_row].answer.values).sum()
    print("Consistency in flipped triplets:", inv_responses / len(orig_data))
    return inv_responses / len(orig_data)


def get_1d_scaling(subj,data):
    objects, queries = preprocess_queries(data)
    soe = cbl.embedding.SOE(1)
    X = soe.fit_transform(queries)
    print("train acc", soe.score(queries))

    result = {'subj': subj,
            'sph': objects['sph'],
            'add': objects['add'],
            'scaling': X[:, 0]}
    return pd.DataFrame(result), soe.score(queries)


def _align_embeddings(embeddings, return_disparity=True, standardize_rotation=True):
    from scipy.linalg import orthogonal_procrustes
    from sklearn.utils import check_array
    from sklearn.utils import resample
    from sklearn.base import clone
    from joblib import Parallel, delayed

    reference = check_array(embeddings[0], copy=True)
    others = [check_array(e, copy=True) for e in embeddings[1:]]

    # standardize the reference embedding: translation, scale, rotation
    reference -= reference.mean()
    reference /= np.linalg.norm(reference)
    if standardize_rotation:
        U, S, _ = np.linalg.svd(reference, full_matrices=False)
        # flip sign based on absolute value for deterministic results
        max_abs = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[max_abs, range(U.shape[1])])
        reference = U * S * signs

    # align others translation, scale, rotation
    for other in others:  # inplace operations
        other -= other.mean()
        other /= np.linalg.norm(other)
        R, scale = orthogonal_procrustes(reference, other)
        other[:] = scale * (other @ R.T)

    if return_disparity:
        disparities = np.array([((reference - other)**2).sum() for other in others])
        return np.array([reference] + others), disparities
    else:
        return np.array([reference] + others)


def align_scales(df):
    """ Align scales with procrustes analysis.  """
    wide_df = pd.pivot(df, columns=['sph', 'add'], index=['subj'])
    zero_index = wide_df.columns.get_loc(('scaling', 0.0, 0.0))
    max_index = wide_df.columns.get_loc(('scaling', 5.0, 3.0))

    scales, disp = _align_embeddings(wide_df.to_numpy().reshape(*wide_df.shape, -1))
    mean_scale = scales.mean(axis=0)
    if mean_scale[0] > mean_scale[-1]:
        scales = -scales
    scales = (scales - mean_scale[zero_index]) / (mean_scale[max_index] - mean_scale[zero_index])

    wide_df = pd.DataFrame(scales.reshape(wide_df.shape), columns=wide_df.columns, index=wide_df.index)
    return wide_df.stack(level=[1, 2]).reset_index()


def main(subj_list):
    for subj in subj_list:
        print("Starting with subject: " + str(subj))
        data = get_trial_df(subj)
        result, emb_acc = get_1d_scaling(subj, data)


        if os.path.isfile(DATA_DIR / 'scaling.pkl'):
            df = pd.read_pickle(DATA_DIR / 'scaling.pkl')
            # remove the current subject from df, to overwrite existing analysis
            df = df.drop(df[df.subj == subj].index)
            df = pd.concat([df, pd.DataFrame(result)])
        else:
            df = result
        df = align_scales(df)
        df.to_pickle(DATA_DIR / 'scaling.pkl')

        ### Save scores: trivial trial performance and embedding accuracy
        perform = get_performance(data)
        consistency = get_flipped_consistency(data)
        scores = pd.DataFrame({'subj': [subj],
                  'emb_acc': [emb_acc],
                  'trivial_perform': [perform],
                  'consistency': [consistency]})
        if os.path.isfile(DATA_DIR / 'scores.pkl'):
            df = pd.read_pickle(DATA_DIR / 'scores.pkl')
            # remove the current subject from df, to overwrite existing analysis
            df = df.drop(df[df.subj == subj].index)
            df = pd.concat([df,scores])
        else:
            df = scores
        df.to_pickle(DATA_DIR / 'scores.pkl')


SUBJ_LIST = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SUBJ_NAME = {3: '1', 4: '2', 5: '3', 7: '4', 8: '5',
             9: '6', 10: '7', 11: '8', 12: '9',
             13: '10', 14: '11', 15: '12', 16: '13'}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        subj_list = list(map(int, sys.argv[1].split(',')))
    else:
        subj_list = SUBJ_LIST
    main(subj_list)