import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import time

# Model Parameters
a = 1      # Prey-Prey Linear Long-Range Attraction Gain
b = 0.2    # Prey-Predator Repulsion Strength
c = 2.5    # Pred-Prey Attraction Strength
d = 0    # Pred-Pred Short Range Repulsion Gain
e = 1      # Pred-Pred Linear Long-Range Attraction Gain
p = 3      # Power law for Predator-Prey Interactions

# Simulation Parameters
T = 1000   # Number of timesteps to run the model for (None = run forever)
dt = 0.01   # 'length' of each timestep in the simulation
W = 2      # Width of Field
H = 2      # Height of Field

# Prey parameters
N = 500    # Number of Prey

# Predator Parameters
M = 1      # Number of Predators

# Visualization Parameters
fps = 30    # Number of frames to display per second
ticks_per_frame = 6


def run_model():
    frames = []
    
    min_r = 0
    max_r = W/2
    # Randomize Intitial Prey Positions
    init_theta = np.random.rand(N,1)*6.28
    init_rad = (np.random.rand(N, 1)*(max_r-min_r))+min_r
    prey_pos = np.hstack((np.cos(init_theta)*init_rad, np.sin(init_theta)*init_rad))
    # Initialize Prey Velocitites
    prey_v = np.random.rand(N, 2)
    
    # Place Predator in Cemter of Field
    init_theta = np.random.rand(M,1)*6.28
    init_rad = (np.random.rand(M, 1)*(max_r-min_r))+min_r
    pred_pos = np.hstack((np.cos(init_theta)*init_rad, np.sin(init_theta)*init_rad))
    
    # Initialize Predator Velocities
    pred_v = np.random.rand(M, 2)
    
    # Initialize Force Arrays       
    F_prey_prey = np.zeros((N, 2))          # sum of forces of ith prey w.r.t. all other prey in x-direction
    F_prey_pred = np.zeros((N, 2))          # sum of forces of the ith prey w.r.t all predators in x-direction
    
    F_pred_pred = np.zeros((M, 2))          # sum of forces of the ith predator w.r.t all predators in x-direction
    F_pred_prey = np.zeros((M, 2))          # sum of forces of the ith predator w.r.t all prey in x-direction

    # Create the Figure
    fig, ax = plt.subplots()

    # Loop for Simulation
    i = 0
    avg_tps = 0
    start_time = time.time()
    while i < T or T is None:   
        # plot in real time
        if ticks_per_frame is not None:
            if i%ticks_per_frame == 0:
                prey_v_norm = np.linalg.norm(prey_v, axis=1)
                prey_q = ax.quiver(prey_pos[:,0], prey_pos[:,1], prey_v[:,0]/prey_v_norm, prey_v[:,1]/prey_v_norm)
                
                pred_v_norm = np.linalg.norm(pred_v, axis=1)
                pred_q = ax.quiver(pred_pos[:,0], pred_pos[:,1], pred_v[:,0]/pred_v_norm, pred_v[:,1]/pred_v_norm, color='red')
                
                title = ax.text(0.5,1.05,f"{round(i*dt,3)}", 
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes)
                # plt.pause(0.001)
                frames.append([title, prey_q, pred_q])
                    
        #Calculate Force Arrays
        for j in range(N):
            # Force between this Prey and All other prey in the X and Y
            d_py_py = prey_pos[j,:] - prey_pos[np.arange(N)!=j, :]    #(N-1) x 1 array of x distance from prey j to all other prey 
            d_py_py_norm = np.power(np.linalg.norm(d_py_py, axis=1), 2).reshape(-1,1)
            F_prey_prey[j,:] = np.mean((d_py_py/d_py_py_norm)-(a*d_py_py), axis=0)
            
            # Force between this Prey and All Predators in the X and Y
            d_py_pd = prey_pos[j,:] - pred_pos  #(M) x 1 array of x distance from prey j to all predators 
            d_py_pd_norm = np.power(np.linalg.norm(d_py_pd, axis=1), 2).reshape(-1,1)
            F_prey_pred[j,:] = b*np.mean(d_py_pd/d_py_pd_norm, axis=0)
            
        for j in range(M):
            # Force between this Predator and All Prey in the X and Y
            d_pd_py = prey_pos - pred_pos[j,:]
            d_pd_py_norm = np.power(np.linalg.norm(d_pd_py, axis=1), p).reshape(-1,1)
            F_pred_prey[j,:] = c*np.mean(d_pd_py/d_pd_py_norm, axis=0)
            
            # Force between this Prey and All other prey in the X and Y (if more than one predator)
            if M > 0:
                d_pd_pd = pred_pos[j,:] - pred_pos[np.arange(M)!=j, :]    #(N-1) x 1 array of x distance from prey j to all other prey 
                d_pd_pd_norm = np.power(np.linalg.norm(d_pd_pd, axis=1), 2).reshape(-1,1)
                F_pred_pred[j,:] = np.mean((d*d_pd_pd/d_pd_pd_norm)-(e*d_pd_pd), axis=0)
            else:
                F_pred_pred[j,:] = np.zeros((1,2))


        
        # Change the Velocity using Forces
        prey_v = F_prey_prey + F_prey_pred
        
        # Change the Predator Velocity using Forces
        pred_v = F_pred_prey + F_pred_pred
        
        # Clamp Magnitude of Prey and Predator Velocities to prevent explosions
        prey_v_norm = np.linalg.norm(prey_v, axis=1).reshape(-1,1)
        prey_v_norm_clip = np.clip(prey_v_norm, -W/8, W/8) # Clamp the magnitude of the velocity to maintain direction
        prey_v = prey_v/prey_v_norm*prey_v_norm_clip
        
        pred_v_norm = np.linalg.norm(pred_v, axis=1).reshape(-1,1)
        pred_v_norm_clip = np.clip(pred_v_norm, -W/8, W/8)
        pred_v = pred_v/pred_v_norm*pred_v_norm_clip
        
        # Update Prey Location
        prey_pos += dt*prey_v
        
        # Update Predator Location
        pred_pos += dt*pred_v
        
        # Clamp Prey and Predator Location to Field
        prey_pos[:,0] = np.clip(prey_pos[:,0], -W, W)
        prey_pos[:,1] = np.clip(prey_pos[:,1], -H, H)
        pred_pos[:,0] = np.clip(pred_pos[:,0], -W, W)
        pred_pos[:,1] = np.clip(pred_pos[:,1], -H, H)
        
        
        
        i+=1
        end_time = time.time()
        avg_tps = (avg_tps*(i-1)/i) + ((end_time-start_time)/i)
        start_time = end_time
        if i%10 == 0:
            print(f"iteration {i}/{T}, time remaining = {avg_tps*(T-i):0,.2f} sec          ", end="\r")
    print("")
    
    if ticks_per_frame is not None:
        # Convert the list of frames into an animation
        anim = ani.ArtistAnimation(fig=fig, artists=frames, interval=int(1000/fps), blit=True, repeat=False)
        # Save the animation as an mp4 file
        print(f"Exporting Animation: {len(frames)} frames @ {fps} fps")
        export_start = time.time()
        plt.show()
        anim.save('media/animation.mp4', writer='ffmpeg', fps=fps, dpi=100)
        print(f"Export Finished, took {time.time()-export_start} sec")
  

if __name__ == "__main__":
    np.random.seed(69)
    run_model()