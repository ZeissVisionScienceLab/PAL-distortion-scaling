import numpy as np
import pandas as pd
import os
from pathlib import Path
import sys
module_path = os.path.abspath('./gaze3d')
sys.path.insert(0, module_path)
import gaze3d as tracking
import matplotlib.pyplot as plt

def read_all_tracking_files(subj):
    # the first repetition tracking file should always exist:
    print("Reading file" , str(subj).zfill(3)  + '_measurement_tracking.csv')
    df = tracking.read_measurement(Path('./measurements' , str(subj).zfill(3), str(subj).zfill(3)  + '_measurement_tracking.csv'))
    if df is not None:
        df['session'] = 0
    # now check for more tracking files
    # first find for number of tracking files
    max_session = 0
    for file in os.listdir(Path('./measurements/' , str(subj).zfill(3))):
        if file.startswith(str(subj).zfill(3)) & ('tracking' in file):
            try:
                session = int(file.split('_')[2]) # extract session from filename
                if session > max_session:
                    max_session = session
            except:
                continue
    print('Number of sessions found: ' + str(max_session))
    for session in range(1,max_session+1):
        file = Path('./measurements', str(subj).zfill(3), str(subj).zfill(3)  + '_measurement_' + str(session).zfill(2) + '_tracking.csv')
        if not os.path.exists(file):
            print('Session ' + file + 'not found')
        print("Reading file" , file)
        df_tmp = tracking.read_measurement(file)
        if df_tmp is not None:
            df_tmp['session'] = session
            df = pd.concat([df, df_tmp]) # combine in one dataframe
    return df


# get number of trials for allocating start_trial and end_trial variables
#TODO: add last message for end of last trial??
def get_trial_index(data):
    messages = data.msg[~data.msg.isnull()]
    n_trials = int(messages.iloc[-1].lower().split("trial")[1].split("stim")[0]) + 1
    print('Trials found: ' + str(n_trials))
    ind = messages.index
    start_index = [[[],[],[]] for _ in range(n_trials)]
    end_index = [[[],[],[]] for _ in range(n_trials)]
    rep_start = [0 for _ in range(n_trials)]
    rep_end = [0 for _ in range(n_trials)]
    print('Total messages: ' + str(len(messages)))
    for i in range(len(messages)):
        trial = int(messages.iloc[i].lower().split("trial")[1].split("stim")[0])
        stiml = int(messages.iloc[i].lower().split("trial")[1].split("stim")[1])
        print(trial)
        print(stiml)
        if('start' in messages.iloc[i].lower()):
            start_index[trial][stiml].append(ind[i])
            rep_start[trial]+=1
        if('end' in messages.iloc[i].lower()):
            end_index[trial][stiml].append(ind[i])
            rep_end[trial]+=1

    return start_index,end_index,n_trials, rep_start, rep_end

def main(subj_list):
    for subj in subj_list:
        print("Starting with subject: " + str(subj))

        data = read_all_tracking_files(subj)
        # Reset index to have unique index
        data = data.reset_index()

        start_index, end_index, n_trials, rep_start, rep_end = get_trial_index(data)

        # Do some sanity checks on start and end indeces
        for t in range(n_trials):
            for s in range(3):
                if len(end_index[t][s]) != len(start_index[t][s]):
                    print("Different reps for start and end index")
                    print(str(len(start_index[t][s])) + ' vs ' + str(len(end_index[t][s])))
                    print('T: ' + str(t) + ' S: ' + str(s))
                    #TODO: add index manually by checking timestamp
                    continue
                if len(end_index[t][s]) == 0:
                    print("Zero reps found")
                    continue
                for rep in range(int(rep_start[t]/3)):
                    if end_index[t][s][rep] - start_index[t][s][rep] <=0:
                        print('error')
                        print('T: ' + str(t) + ' S: ' + str(s) + ' R: ' + str(rep))

        # check NaNs in gaze before filtering
        print('NaN in gaze left: ' + str(data['left_gaze_x'].isna().sum()))
        print('NaN in gaze right: ' + str(data['right_gaze_x'].isna().sum()))
        print('NaN in gaze comb: ' + str(data['comb_gaze_x'].isna().sum()))
        # first mask trials with gaze vec = (0,0,1)
        ind_l = ((data.left_gaze_x == 0) & (data.left_gaze_y == 0))
        ind_r = ((data.right_gaze_x == 0) & (data.right_gaze_y == 0))
        ind_comb = ((data.comb_gaze_x == 0) & (data.comb_gaze_y == 0))
        print('Invalid samples left: ' + str(ind_l.sum()))
        print('Invalid samples right: ' + str(ind_r.sum()))
        print('Invalid samples comb: ' + str(ind_comb.sum()))
        # set samples to NaN for all gaze related measurements
        cols = ['left_gaze_x', 'left_gaze_y', 'left_gaze_z',
                'right_gaze_x', 'right_gaze_y', 'right_gaze_z',
                'comb_gaze_x', 'comb_gaze_y', 'comb_gaze_z',
                'ooi',
                'pof_x', 'pof_y', 'pof_z']
        data.loc[ind_l, cols] = np.nan
        data.loc[ind_r, cols] = np.nan
        data.loc[ind_comb, cols] = np.nan
        # check NaNs in gaze after filtering
        print('After filtering')
        print('NaN in gaze left: ' + str(data['left_gaze_x'].isna().sum()))
        print('NaN in gaze right: ' + str(data['right_gaze_x'].isna().sum()))
        print('NaN in gaze comb: ' + str(data['comb_gaze_x'].isna().sum()))

        rep1 = []
        rep2 = []
        for trial in range(n_trials):
            # for stiml in range(3):
            rep1.append(len(start_index[trial][0]))
            rep2.append(len(end_index[trial][2]))
        # Get indices of trials where rep of stat of stim 0 is more often then end of stim 2 (unfinished trials)
        ind =[i for i,x in enumerate([(a_i - b_i != 0) for a_i, b_i in zip(rep1, rep2)]) if x]
        for i in ind:
            print('trial:' , i)
            print('stim:' , 0)
            print(start_index[i][0])
            print(end_index[i][0])
            print('stim:' , 1)
            print(start_index[i][1])
            print(end_index[i][1])
            print('stim:' , 2)
            print(start_index[i][2])
            print(end_index[i][2])

        # remove unfinished trial (session stopped before end)
        for i in ind:
            print('trial: ' + str(i))
            for s in range(3):
                print('stim: ' + str(s))
                if len(start_index[i][s]) < len(end_index[i][s]):
                    print('More end than start')
                if len(start_index[i][s]) > len(end_index[i][s]):
                    print('More start than end')
                    # first check if there is a started rep inbetween without end
                    for r in range(len(end_index[i][s])):
                        print(r)
                        if end_index[i][s][r] > start_index[i][s][r + 1]:
                            print('Rep ' + str(r) + 'was not finished')
                            del (start_index[i][s][r])
                            if (len(start_index[i][s]) == len(end_index[i][s])):
                                break  # same reps now for start and end -> break out of for loop,
                    # if start and end still different length, then add end index for last repetition
                    if len(start_index[i][s]) > len(end_index[i][s]):
                        end_index[i][s].append(start_index[i][s][
                                                   -1] + 1)  # just take the next index, we can not be sure how long the presentation was
            # as a final step we still have to check if all stimuli now have equal reps
            for s in [2, 1]:
                if len(start_index[i][s]) < len(end_index[i][s - 1]):  # previous stim has more reps
                    print("Stim " + str(s - 1) + " has more reps than " + str(s))
                    # find the correct rep to delete
                    for r in range(len(start_index[i][s])):
                        if start_index[i][s][r] > end_index[i][s - 1][r + 1]:
                            del (start_index[i][s - 1][r])
                            del (end_index[i][s - 1][r])
                            if s == 2:
                                del (start_index[i][s - 2][r])
                                del (end_index[i][s - 2][r])
                        # check if reps of the different stimuli fit now:
                        if len(start_index[i][s]) == len(end_index[i][s - 1]):
                            # length fits now, we can stop
                            break


        #new dataframe only containing the tracking data during trials
        data['t'] = np.nan
        data['s'] = np.nan
        data['rep'] = np.nan

        data_filtered = pd.DataFrame()

        for t in range(n_trials):
            for rep in range(len(start_index[t][0])):
                print('t:' + str(t) + 'rep:' + str(rep))
                data.loc[start_index[t][0][rep]:end_index[t][2][rep],
                't'] = t  # set trial also for samples between stimuli (during short fading)
                data.loc[start_index[t][0][rep]:end_index[t][2][rep], 'rep'] = rep
                for s in range(3):
                    data.loc[start_index[t][s][rep]:end_index[t][s][rep], 's'] = s  # stimulus is set not for fading phases
                    if len(end_index[t][s]) != len(start_index[t][s]):
                        print('T: ' + str(t) + ' S: ' + str(s) + ' r: ' + str(rep))
                        print("Different reps for start and end index")
                        print(str(len(start_index[t][s])) + ' vs ' + str(len(end_index[t][s])))

                data_filtered = pd.concat([data_filtered, data.loc[start_index[t][0][rep]:end_index[t][2][rep]]])

        data_filtered.sort_index(inplace=True)

        # calculate yaw pitch and roll from camera orientation

        y, p, r = tracking.quat2ypr(data_filtered[['cam_rot_x','cam_rot_y','cam_rot_z','cam_rot_w']])
        data_filtered['yaw'] = y
        data_filtered['pitch'] = p
        data_filtered['roll'] = r
        data_filtered['dt'] = data_filtered.Timestamp.diff()
        ind = (data_filtered.dt == 0)
        data_filtered.loc[ind,'dt'] = np.nan
        data_filtered['dyawdt'] = data_filtered.yaw.diff() / data_filtered.dt
        data_filtered['dpitchdt'] = data_filtered.pitch.diff() / data_filtered.dt
        data_filtered['drolldt'] = data_filtered.roll.diff() / data_filtered.dt

        #TODO: also add presentation based compiling (for each distortino presentation)
        trial_data = pd.DataFrame(columns=['trial', 'mean_yaw_vel', 'mean_pitch_vel','mean_roll_vel',
                                           'gaze_ground','gaze_ceiling','gaze_walls'])

        trial_data = pd.DataFrame(columns=['trial', 'rep',
                                           'mean_yaw_vel', 'mean_pitch_vel', 'mean_roll_vel',
                                           'yaw_ampl', 'pitch_ampl', 'roll_ampl',
                                           'yaw_freq', 'pitch_freq', 'roll_freq',
                                           'gaze_ground', 'gaze_ceiling', 'gaze_walls',
                                           'mean_l_gaze_x', 'mean_l_gaze_y', 'mean_l_gaze_z',
                                           'mean_r_gaze_x', 'mean_r_gaze_y', 'mean_r_gaze_z',
                                           'mean_c_gaze_x', 'mean_c_gaze_y', 'mean_c_gaze_z',
                                           'mean_c_gaze_long' ,'mean_c_gaze_lat'])


        def get_peak_oscillation(time, y):
            from scipy.fft import fft, fftfreq
            yf = fft(y - y.mean())
            N = len(y)
            xf = fftfreq(N, (time[-1] - time[0]) / N)
            # remove the negative frequencies
            xf = xf[:N // 2]
            yf = yf[:N // 2]
            peak_freq = xf[abs(yf) == abs(yf).max()]
            amplitude = 2.0 / N * abs(yf).max()
            return peak_freq[0], amplitude


        i = 0
        for t in range(n_trials):
            for rep in range(len(start_index[t][0])):
                print("t: " + str(t) + " r: " + str(rep))
                # get index for current trial and rep
                index = (data_filtered.t == t) & (data_filtered.rep == rep)
                # calculate mean head movement velocity
                yaw_vel = data_filtered.dyawdt[index].abs().mean()
                roll_vel = data_filtered.drolldt[index].abs().mean()
                pitch_vel = data_filtered.dpitchdt[index].abs().mean()

                # perform FFT to check head oscillations
                time = data_filtered.Timestamp[index].values.flatten()
                yaw = data_filtered.yaw[index].values.flatten()
                pitch = data_filtered.pitch[index].values.flatten()
                roll = data_filtered.roll[index].values.flatten()
                yaw_freq, yaw_ampl = get_peak_oscillation(time, yaw)
                pitch_freq, pitch_ampl = get_peak_oscillation(time, pitch)
                roll_freq, roll_ampl = get_peak_oscillation(time, roll)

                # gaze distribution in scene
                n_samples = data_filtered.ooi[index].dropna().count()
                if n_samples == 0:
                    print('No samples in trial ' + str(t) + ' rep ' + str(rep))
                    ground = np.nan
                    walls = np.nan
                    ceiling = np.nan
                else:
                    ground = (data_filtered.ooi[index] == 'Ground').dropna().sum() / n_samples
                    walls = (data_filtered.ooi[index] == 'Walls').dropna().sum() / n_samples
                    ceiling = (data_filtered.ooi[index] == 'Ceiiling').dropna().sum() / n_samples

                ## mean gaze direction in FoV
                # take sample mean
                mean_l_gaze_x = data_filtered.left_gaze_x[index].mean()
                mean_l_gaze_y = data_filtered.left_gaze_y[index].mean()
                mean_l_gaze_z = data_filtered.left_gaze_z[index].mean()

                mean_r_gaze_x = data_filtered.right_gaze_x[index].mean()
                mean_r_gaze_y = data_filtered.right_gaze_y[index].mean()
                mean_r_gaze_z = data_filtered.right_gaze_z[index].mean()

                mean_c_gaze_x = data_filtered.comb_gaze_x[index].mean()
                mean_c_gaze_y = data_filtered.comb_gaze_y[index].mean()
                mean_c_gaze_z = data_filtered.comb_gaze_z[index].mean()

                _, mean_long_comb, mean_lat_comb = tracking.cart2geographic(mean_c_gaze_x,
                                                                            mean_c_gaze_y,
                                                                            mean_c_gaze_z)

                trial_data.loc[i] = [t, rep,
                                     yaw_vel, pitch_vel, roll_vel,
                                     yaw_ampl, pitch_ampl, roll_ampl,
                                     yaw_freq, pitch_freq, roll_freq,
                                     ground, ceiling, walls,
                                     mean_l_gaze_x, mean_l_gaze_y, mean_l_gaze_z,
                                     mean_r_gaze_x, mean_r_gaze_y, mean_r_gaze_z,
                                     mean_c_gaze_x, mean_c_gaze_y, mean_c_gaze_z,
                                     mean_long_comb,mean_lat_comb]
                i += 1

        # Head-relative gaze distribution
        # calculate longitude and latitude

        r_left, long_left, lat_left = tracking.cart2geographic(data_filtered.left_gaze_x,data_filtered.left_gaze_y,data_filtered.left_gaze_z)
        r_right, long_right, lat_right = tracking.cart2geographic(data_filtered.right_gaze_x,data_filtered.right_gaze_y,data_filtered.right_gaze_z)
        r_comb, long_comb, lat_comb = tracking.cart2geographic(data_filtered.comb_gaze_x,data_filtered.comb_gaze_y,data_filtered.comb_gaze_z)

        data_filtered['r_left'] = r_left
        data_filtered['long_left'] = long_left
        data_filtered['lat_left'] = lat_left

        data_filtered['r_right'] = r_right
        data_filtered['long_right'] = long_right
        data_filtered['lat_right'] = lat_right

        data_filtered['r_comb'] = r_comb
        data_filtered['long_comb'] = long_comb
        data_filtered['lat_comb'] = lat_comb

        ## Scene relative gaze distribution
        cam_pos = [0, 1.5, 1]
        x_rel = data_filtered.pof_x-cam_pos[0]
        y_rel = data_filtered.pof_y-cam_pos[1]
        z_rel = data_filtered.pof_z-cam_pos[2]
        x_img = x_rel/z_rel
        y_img = y_rel / z_rel

        #r_world, long_world,lat_world = tracking.cart2geographic(data_filtered.pof_x-cam_pos[0],
        #                                    data_filtered.pof_y-cam_pos[1],
        #                                    data_filtered.pof_z-cam_pos[2])
        data_filtered['x_img'] = x_img
        data_filtered['y_img'] = y_img
        #data_filtered['lat_world'] = lat_world

        # Saving
        data.to_pickle('./data/' + str(subj).zfill(3) + 'tracking.pkl')
        data_filtered.to_pickle('./data/' + str(subj).zfill(3) + 'filtered_tracking.pkl')
        trial_data.to_pickle('./data/' + str(subj).zfill(3) + 'trial_data.pkl')

        ## Create "overview" dataframe for subject
        _, long_mean, lat_mean = tracking.cart2geographic(data_filtered.comb_gaze_x.mean(),data_filtered.comb_gaze_y.mean(),data_filtered.comb_gaze_z.mean())

        behaviour = pd.DataFrame({'subj': [subj],
                                  'mean_yaw': [data_filtered.dyawdt.abs().mean()],
                                  'mean_pitch': [data_filtered.dpitchdt.abs().mean()],
                                  'mean_roll': [data_filtered.drolldt.abs().mean()],
                                  'mean_gaze_long': [long_mean],
                                  'mean_gaze_lat': [lat_mean],
                                  'std_gaze_long': [data_filtered.long_comb.std()],
                                  'std_gaze_lat': [data_filtered.lat_comb.std()]})

        if os.path.isfile('./data/behaviour.pkl'):
            df = pd.read_pickle('./data/behaviour.pkl')
            # remove the current subject from df, to overwrite existing analysis
            df = df.drop(df[df.subj == subj].index)
            df = pd.concat([df, pd.DataFrame(behaviour)])
        else:
            df = behaviour
        df.to_pickle('./data/behaviour.pkl')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        subj_list = list(map(int, sys.argv[1].split(',')))
    else:
        subj_list = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    main(subj_list)
