# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# See read the https://floris.readthedocs.io for documentation


import matplotlib.pyplot as plt
import floris.tools as wfct
import floris.tools.visualization as vis
import floris.tools.cut_plane as cp
from floris.tools.optimization import YawOptimizationWindRoseParallel
import floris.tools.wind_rose as rose
import floris.tools.power_rose as pr
import numpy as np
import pandas as pd
import WakeSteering_US.cp_for_any_turb as cturb

if __name__ == '__main__':
    
        # Instantiate the FLORIS object
    fi = wfct.floris_utilities.FlorisInterface("example_input.json")
    
    # Define wind farm coordinates and layout
    mf =pd.read_pickle('Wind_US_Database')
    kf = (mf.loc[mf['p_name'] == 'Cisco'])
    wf_coordinate = [kf["ylat"].mean(),kf["xlong"].mean()]
    
    # Set wind farm to N_row x N_row grid with constant spacing 
    # (2 x 2 grid, 5 D spacing)
    D = fi.floris.farm.turbines[0].rotor_diameter
    lat_y = kf['ylat'].values
    long_x = kf['xlong'].values
    
    layout_x, layout_y= nf.longlat_to_utm(lat_y, long_x)
    
    N_turb = len(layout_x)
    
    fi.reinitialize_flow_field(layout_array=(layout_x, layout_y), wind_direction=270.0, wind_speed=8.0)
    fi.calculate_wake()
    
    #Diameter and Rated power based on the wind farm
    D = kf["t_rd"]
    P_r = kf["t_cap"]
    hub_h = kf["t_hh"]
    
    C_p_rated = 0.43003137
    C_t_rated = 0.70701647
    
    #Normalized wind speed for any turbine
    tf= pd.read_pickle('Wind_Cp_look_up_table')
    
    ## Enumerate so for each turbine 
    for count, turbine in enumerate(fi.floris.farm.flow_field.turbine_map.turbines):
            turbine.rotor_diameter = D.iloc[count]
            turbine.hub_height = hub_h.iloc[count]
            T_Area = (np.pi* (D.iloc[count]**2)) /4
            U_turb_rated= (2* P_r.iloc[count]*(10**3)/ (C_p_rated * 1.225* T_Area))**(1/3)
            U_turb_norm =  tf.iloc[:,0] / U_turb_rated
            cp_new = cturb.cp_for_any_turb(U_turb_norm)
            ct_new = cturb.ct_for_any_turb(U_turb_norm)
            turbine.power_thrust_table["power"] = cp_new
            turbine.power_thrust_table["thrust"] = ct_new
    
    # set min and max yaw offsets for optimization 
    min_yaw = 0.0
    max_yaw = 25.0
    
    # Define minimum and maximum wind speed for optimizing power. 
    # Below minimum wind speed, assumes power is zero.
    minimum_ws = 3.0
    maximum_ws = 15.0

    # ================================================================================
    print('Plotting the FLORIS flowfield...')
    # ================================================================================

    # Initialize the horizontal cut
    hor_plane = wfct.cut_plane.HorPlane(
        fi.get_flow_data(),
        fi.floris.farm.turbines[0].hub_height
    )

    # Plot and show
    fig, ax = plt.subplots()
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    ax.set_title('Baseline flow for U = 8 m/s, Wind Direction = 270$^\circ$')

    # ================================================================================
    print('Importing wind rose data...')
    # ================================================================================

    # Create wind rose object and import wind rose dataframe using WIND Toolkit HSDS API.
    # Alternatively, load existing .csv file with wind rose information.
    calculate_new_wind_rose = True

    wind_rose = rose.WindRose()

    if calculate_new_wind_rose:

    	wd_list = np.arange(0,360,5)
    	ws_list = np.arange(0,26,1)

    	df = wind_rose.import_from_wind_toolkit_hsds(wf_coordinate[0],
    	                                                    wf_coordinate[1],
    	                                                    ht = 100,
    	                                                    wd = wd_list,
    	                                                    ws = ws_list,
    	                                                    limit_month = None,
    	                                                    st_date = None,
    	                                                    en_date = None)

    else:
    	df = wind_rose.load('windtoolkit_geo_center_us.p')

    # plot wind rose
    wind_rose.plot_wind_rose()

    # =============================================================================
    print('Finding baseline and optimal wake steering power in FLORIS...')
    # =============================================================================

    # Instantiate the parallel optimization object
    yaw_opt = YawOptimizationWindRoseParallel(fi, df.wd, df.ws, df.ti,
                                       minimum_yaw_angle=min_yaw, 
                                       maximum_yaw_angle=max_yaw,
                                       minimum_ws=minimum_ws,
                                       maximum_ws=maximum_ws)

    # Perform optimization
    df_base, df_opt = yaw_opt.optimize()

    # Summarize using the power rose module
    power_rose = pr.PowerRose()
    case_name = 'Example '+kf['p_name'].iloc[0]+ ' Wind Farm'

    # combine wind farm-level power into one dataframe
    df_power = pd.DataFrame({'ws':df.ws,'wd':df.wd, \
        'freq_val':df.freq_val,'power_no_wake':df_base.power_no_wake, \
        'power_baseline':df_base.power_baseline,'power_opt':df_opt.power_opt})

    # initialize power rose
    df_yaw = pd.DataFrame([list(row) for row in df_opt['yaw_angles']],columns=[str(i) for i in range(1,N_turb+1)])
    df_yaw['ws'] = df.ws
    df_yaw['wd'] = df.wd
    df_turbine_power_no_wake = pd.DataFrame([list(row) for row in df_base['turbine_power_no_wake']],columns=[str(i) for i in range(1,N_turb+1)])
    df_turbine_power_no_wake['ws'] = df.ws
    df_turbine_power_no_wake['wd'] = df.wd
    df_turbine_power_baseline = pd.DataFrame([list(row) for row in df_base['turbine_power_baseline']],columns=[str(i) for i in range(1,N_turb+1)])
    df_turbine_power_baseline['ws'] = df.ws
    df_turbine_power_baseline['wd'] = df.wd
    df_turbine_power_opt = pd.DataFrame([list(row) for row in df_opt['turbine_power_opt']],columns=[str(i) for i in range(1,N_turb+1)])
    df_turbine_power_opt['ws'] = df.ws
    df_turbine_power_opt['wd'] = df.wd

    power_rose.initialize(case_name, df_power, df_yaw, df_turbine_power_no_wake, df_turbine_power_baseline, df_turbine_power_opt)

    fig, axarr = plt.subplots(3, 1, sharex=True, figsize=(6.4, 6.5))
    power_rose.plot_by_direction(axarr)
    power_rose.report()

    plt.show()
