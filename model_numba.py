import time
import numpy as np
from numba import jit
import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as ani

default_sim = {
    "T":500,
    "dt":0.1,
    "W":2,
    "H":2
}

default_vis = {
    "fps": 12,
    "ticks_per_frame":12,
    "show_animation":True,
    "export_animation":True,
    "animation_path": "models/animation.mp4",
    "animation_codec": "libx264"
}


@jit(nopython=True)
def norm(array, axis=0):
    return np.sqrt(np.sum(np.power(array, 2), axis=axis))


@jit(nopython=True)
def mean(array, axis=0):
    return np.sum(array, axis=axis)/np.shape(array)[axis]

def render_frame( ax, i, dt, prey_pos, prey_v, pred_pos, pred_v):
    """Renders all the Artists for displaying the model

    Args:
        ax (plt.Axes): Current Matplotlib Axis
        i (int): Current Iteration
        prey_pos (np.ndarray[Any]): Nx2 array of the prey's XY position
        prey_v (np.ndarray[Any]): Nx2 array of the prey's XY velocities
        pred_pos (np.ndarray[Any]): Mx2 array of the predators's XY position
        pred_v (np.ndarray[Any]): Mx2 array of the predators's XY velocity

    Returns:
        List[plt.Artist]: Artists used in the current frame
    """
    ax.clear()
    # Plot Prey Positions/Velocities
    prey_v_norm = norm(prey_v, axis=1)
    prey_q = ax.quiver(
        prey_pos[:, 0], prey_pos[:, 1], prey_v[:, 0]/prey_v_norm, prey_v[:, 1]/prey_v_norm)

    # Plot Predator Position/Velocities
    pred_v_norm = norm(pred_v, axis=1)
    pred_q = ax.quiver(
        pred_pos[:, 0], pred_pos[:, 1], pred_v[:, 0]/pred_v_norm, pred_v[:, 1]/pred_v_norm, color='red')
        
    # min_x = min(np.min(prey_pos[:,0]), np.min(pred_pos[:,0]))
    # max_x = max(np.max(prey_pos[:,0]), np.max(pred_pos[:,0]))
    # min_y = min(np.min(prey_pos[:,1]), np.min(pred_pos[:,1]))
    # max_y = min(np.max(prey_pos[:,1]), np.max(pred_pos[:,1]))

    
    # curr_xlim = ax.get_xlim()
    # ax.set_xlim(min(min_x, curr_xlim[0]), max(max_x, curr_xlim[1]))
    
    # curr_ylim = ax.get_ylim()
    # ax.set_ylim(min(min_y, curr_ylim[0]), max(max_y, curr_ylim[1]))

    # Update Title to Current Timestep
    title = ax.text(0.5, 1.05, f"{round(i*dt,3)}",
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes)
    
    return [title, prey_q, pred_q]#, F_pd_py_q, F_pd_pd_q,]#F_pd_py_qx, F_pd_py_qy, F_pd_pd_qx, F_pd_pd_qy,]

def render_animation(all_prey_pos, all_prey_v, all_pred_pos, all_pred_v, sim_params, vis_params):
    # Parse Simulation Parameters
    T = sim_params["T"]
    dt = sim_params["dt"]
    W = sim_params["W"]
    H = sim_params["H"]
    
    ticks_per_frame = vis_params["ticks_per_frame"]
    fps = vis_params["fps"]
    show_animation = vis_params["show_animation"]
    export_animation = vis_params["export_animation"]
    animation_path = vis_params["animation_path"]
    animation_codec = vis_params["animation_codec"]
    
    fig = plt.figure(figsize=(8,8), dpi=100)
    ax = fig.subplots()
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    def animate(i):
            n = i*ticks_per_frame
            render_frame(ax, n, dt, all_prey_pos[n,:], all_prey_v[n], all_pred_pos[n,:], all_pred_v[n])
    
    animate(0)
            
    if export_animation:
        print(f"Exporting Animation to: {animation_path}")
        canvas_width, canvas_height = fig.canvas.get_width_height()
        cmdstring = ('ffmpeg', 
                    '-y', '-r', str(fps), # overwrite, 1fps
                    '-s', '%dx%d' % (canvas_width, canvas_height), # size of image string
                    '-pix_fmt', 'argb', # format
                    '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
                    '-vcodec', animation_codec, 
                    '-hide_banner', '-loglevel', 'error',
                    animation_path) # output encoding
        p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

        # Draw frames and write to the pipe
        for frame in range(int(T/ticks_per_frame)):
            # draw the frame
            animate(frame)
            fig.canvas.draw()

            # extract the image as an ARGB string
            string = fig.canvas.tostring_argb()

            # write to pipe
            p.stdin.write(string)

        # Finish up
        p.communicate()

    if show_animation:  
        anim = ani.FuncAnimation(fig, animate, repeat=False,
                                    frames=range(int(T/ticks_per_frame)),
                                    interval=int(1000/fps),
                                    blit=True)
        plt.show()
    plt.close(fig=fig)

@jit(nopython=True)
def calculate_forces( prey_pos, pred_pos, a, b, c, d, e, p, N, M):
    """_summary_

    Args:
        prey_pos (np.ndarray[Any]): Nx2 array of the prey's XY position
        pred_pos (np.ndarray[Any]): Mx2 array of the predators's XY position
        a (float): Prey-Prey Long Range Attraction Gain
        b (float): Prey-Predator Short Range Repulsion Gain
        c (float): Predator-Prey Short Range Attraction Gain
        d (float): Predator-Predator Short Range Repulsion Gain
        e (float): Predator-Predator Long Range Attraction Gain
        p (float): Predator-Prey Short Range Attraction Power Law
        N (int): Number of Prey
        M (int): Number of Predators

    Returns:
        Tuple(np.ndarray): F_prey_prey, F_prey_pred, F_pred_prey, F_pred_pred
    """
    # Initialize Force Arrays
    # sum of forces of ith prey w.r.t. all other prey in x-direction
    F_prey_prey = np.zeros((N, 2))
    # sum of forces of the ith prey w.r.t all predators in x-direction
    F_prey_pred = np.zeros((N, 2))

    # sum of forces of the ith predator w.r.t all predators in x-direction
    F_pred_pred = np.zeros((M, 2))
    # sum of forces of the ith predator w.r.t all prey in x-direction
    F_pred_prey = np.zeros((M, 2))

    for j in range(N):
        # Force between this Prey and All other prey in the X and Y
        # (N-1) x 1 array of x distance from prey j to all other prey
        d_py_py = prey_pos[j, :] - prey_pos[np.arange(N) != j, :]
        d_py_py_norm = np.power(norm(d_py_py, axis=1), 2).reshape(-1, 1)
        F_prey_prey[j, :] = mean((d_py_py/d_py_py_norm)-(a*d_py_py), axis=0)

        # Force between this Prey and All Predators in the X and Y
        # (M) x 1 array of x distance from prey j to all predators
        d_py_pd = prey_pos[j, :] - pred_pos
        d_py_pd_norm = np.power(norm(d_py_pd, axis=1), 2).reshape(-1, 1)
        F_prey_pred[j, :] = b*mean(d_py_pd/d_py_pd_norm, axis=0)

    for j in range(M):
        # Force between this Predator and All Prey in the X and Y
        d_pd_py = prey_pos - pred_pos[j, :]
        d_pd_py_norm = np.power(norm(d_pd_py, axis=1), p).reshape(-1, 1)
        F_pred_prey[j, :] = c*mean(d_pd_py/d_pd_py_norm, axis=0)
        
        if M > 1:
            # Force between this Predator and All other predators in the X and Y
            d_pd_pd = pred_pos[j, :] - pred_pos[np.arange(M) != j, :]
            d_pd_pd_norm = np.power(norm(d_pd_pd, axis=1), 2).reshape(-1, 1)
            F_pred_pred[j, :] = mean(d*(d_pd_pd/d_pd_pd_norm)-(e*d_pd_pd), axis=0)
        else:
            F_pred_pred[j, :] = 0

    return F_prey_prey, F_prey_pred, F_pred_prey, F_pred_pred

@jit(nopython=True)
def update_positions(dt, prey_pos, prey_v, pred_pos, pred_v, F_prey_prey, F_prey_pred, F_pred_prey, F_pred_pred):
    # Change the Velocity using Forces
    prey_v = F_prey_prey + F_prey_pred

    # Change the Predator Velocity using Forces
    pred_v = F_pred_prey + F_pred_pred

    # Clamp Magnitude of Prey and Predator Velocities to prevent explosions
    prey_v_norm = norm(prey_v, axis=1).reshape(-1, 1)
    # Clamp the magnitude of the velocity to maintain direction
    prey_v_norm_clip = np.clip(prey_v_norm, -3, 3)
    prey_v = prey_v/prey_v_norm*prey_v_norm_clip

    pred_v_norm = norm(pred_v, axis=1).reshape(-1, 1)
    pred_v_norm_clip = np.clip(pred_v_norm, -3, 3)
    pred_v = pred_v/pred_v_norm*pred_v_norm_clip

    # Update Prey Location
    prey_pos += dt*prey_v

    # Update Predator Location
    pred_pos += dt*pred_v
    
    return prey_pos, prey_v, pred_pos, pred_v

def run_model( a, b, c, d, e, p, N, M, sim_params=default_sim, vis_params=default_vis):
    """Runs the Predator Prey Particle Model wih the provided parameters

    Args:
        a (float): Prey-Prey Long Range Attraction Gain
        b (float): Prey-Predator Short Range Repulsion Gain
        c (float): Predator-Prey Short Range Attraction Gain
        d (float): Predator-Predator Short Range Repulsion Gain
        e (float): Predator-Predator Long Range Attraction Gain
        p (float): Predator-Prey Short Range Attraction Power Law
        N (int): Number of Prey
        M (int): Number of Predators
    """
    print(f"Running Predator-Prey Model with Parameters: \n\t a={a}, b={b}, c={c}, d={d}, e={e}, p={p}, N={N}, M={M}")
    
    # Parse Simulation Parameters
    T = sim_params["T"]
    dt = sim_params["dt"]
    W = sim_params["W"]
    H = sim_params["H"]
    
    ticks_per_frame = vis_params["ticks_per_frame"]
    fps = vis_params["fps"]
    show_animation = vis_params["show_animation"]
    export_animation = vis_params["export_animation"]
    animation_path = vis_params["animation_path"]
    
    
    # Compile List of Frames for Animations and Videos
    frames = []

    # Initialization Parameters
    min_r = 0
    max_r = min(W,H)/2

    # Randomize Intitial Prey Positions
    init_theta = np.random.rand(N, 1)*6.28
    init_rad = (np.random.rand(N, 1)*(max_r-min_r))+min_r
    prey_pos = np.hstack((np.cos(init_theta)*init_rad,
                        np.sin(init_theta)*init_rad))

    # Initialize Prey Velocitites
    prey_v = np.random.rand(N, 2)

    # Place Predator in Cemter of Field
    init_theta = np.random.rand(M, 1)*6.28
    init_rad = (np.random.rand(M, 1)*(max_r-min_r))+min_r
    pred_pos = np.hstack((np.cos(init_theta)*init_rad,
                        np.sin(init_theta)*init_rad))

    # Initialize Predator Velocities
    pred_v = np.random.rand(M, 2)

    # Initialize Accumulators
    all_prey_pos = np.zeros((T,N,2))
    all_prey_v = np.zeros((T,N,2))
    all_pred_pos = np.zeros((T,M,2))
    all_pred_v = np.zeros((T,M,2))   
    
    # Loop for Simulation
    i = 0
    avg_tps = 0
    run_start = time.time()
    start_time = time.time()
    pred_caught_prey = False
    for i in range(T):
        # Calculate Force Arrays
        F_prey_prey, F_prey_pred, F_pred_prey, F_pred_pred = calculate_forces(prey_pos, pred_pos, a, b, c, d, e, p, N, M)

        prey_pos, prey_v, pred_pos, pred_v = update_positions(dt, prey_pos, prey_v, pred_pos, pred_v, F_prey_prey, F_prey_pred, F_pred_prey, F_pred_pred)

        all_prey_pos[i,:,:] = prey_pos
        all_prey_v[i,:,:] = prey_v
        all_pred_pos[i,:,:] = pred_pos
        all_pred_v[i,:,:] = pred_v
        
        # # Clamp Prey and Predator Location to Field
        # prey_pos[:,0] = np.clip(prey_pos[:,0], -W, W)
        # prey_pos[:,1] = np.clip(prey_pos[:,1], -H, H)
        # pred_pos[:,0] = np.clip(pred_pos[:,0], -W, W)
        # pred_pos[:,1] = np.clip(pred_pos[:,1], -H, H)
        
        kill_radius = 0.005
        if i > int(T/10):
            for j in range(M):
                if np.any(norm(prey_pos-pred_pos[j,:],1) < kill_radius):
                    pred_caught_prey = True
                    break

        # Print Status
        end_time = time.time()
        avg_tps = (avg_tps*(i)/(i+1)) + ((end_time-start_time)/(i+1))
        start_time = end_time
        if i % 50 == 0:
            print(f"iteration {i}/{T}, remaining = {avg_tps*(T-(i+1)):.2f}s, elapsed = {time.time()-run_start:.2f}s        ", end="\r")

    # Print Data about Model's Characteristics
    print("")
    
    if export_animation or show_animation:
        render_animation(all_prey_pos, all_prey_v, all_pred_pos, all_pred_v, sim_params, vis_params)
            
    return pred_caught_prey     
    