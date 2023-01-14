# open_vins_tracking

Check the directory open_vins_trajectory
  - Use the tracking_seminar.ipynb
    - Check from code_box 4 { i.e. find_delays_all_modules() }
    - finds all the relevant results 
  - The data files are sorted in the order of the frequency they were executed.
    - For the tum_vi you will find folders named as "tum_vi_descr_20hz.txt" & "tum_vi_20hz.txt", the ones with the "descr" means that 
    I have used ORB method for tracking, and the ones without descr means that KLT(optical flow) has been used for tracking. 
    Open_VINS authors have mentioned that their ORB/feature based tracking isnt working well in all scenarios, so I have skipped for Euroc. 
    
