# =================================================================================================================================================
#                                       Import modules

import matplotlib.pyplot as plt
import Hopf_cl as OSC
import time
import barecmaes2 as cma


def PickleIt(data,fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def GetPickle(fileName):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
    return data

def esStop(es):
    tt=str(time.clock())
    np.save('saved-cma-object'+tt,es)
    os.rename('saved-cma-object'+tt+'.npy','saved-cma-object.npy')

def esResume():
    es=np.load('saved-cma-object.npy')
    return es.item()

def velocity(obj_name,time,co):
    initPos=getInitPos(obj_name)[co]
    curPos=scn.objects[obj_name].worldPosition[co]
    return -(curPos-initPos)/(time+0.01)

bpy.context.scene.game_settings.fps=60.
dt=1./bpy.context.scene.game_settings.fps



# =================================================================================================================================================
#                                       Network creation

#np.random.seed(np.random.randint(0,10000))


# Joints

inpp=[] # Record inputs of servos to be plotted
outp=[] # Record outputs of servos to be plotted 
cont_inp=[] # Reecord control inputs to servos to be plotted
#Joint names
joints=["wrist.L","wrist.R","forearm.L","forearm.R","upper_arm.L","upper_arm.R",\
        "shin_lower.L","shin_lower.R","shin.L","shin.R","thigh.L","thigh.R"]
nJoint=len(joints)
numRec=3 # The number of joint to be plotted

####### Optimization ##############

length_sol=0
sol=[]
iter_num=0
f_sol=[]
name2_es='saved-cma-object.pkl'

# If the opt algorithm hasnt been initialized yet, initialize it
try:
    cma_prop=GetPickle('save_cma')
    initialize_cma=0
except:
    initialize_cma=1


# params to be optimized: w_stance, w_swing and P
if initialize_cma==1:
    #PickleIt(es,'saved-cma-object.pkl')
    initial_param=[93,52,110] # Initial parameters
    sigma0=5 
    es=cma.CMAES(initial_param,sigma0,{'seed':234})
    esStop(es)
    print(es)
    sol=es.ask()
    length_sol=len(sol)
    f_sol=length_sol*[10]
    # Dump [proposed solutions, function values of sol.s, in which iter, initialize_cma]
    PickleIt([sol,f_sol,0,0],'save_cma')
    print('Initialized!')
    
else:
    #es = pickle.load(open(name2_es, 'rb'))
    [sol,f_sol,iter_num,initialize_cma]=GetPickle('save_cma')
    length_sol=len(sol)
    print(iter_num,length_sol)
    if iter_num==length_sol:
        es = esResume()
        iter_num=0
        #es.ask()
        es.tell(sol,f_sol)
        es.disp()
        print(es.best.get())
        sol=es.ask()
        esStop(es)
        PickleIt([sol,length_sol*[10],iter_num,0],'save_cma')
        
    print('Loaded!')
f= lambda x:(x-2)**2

####### Oscillators ###############

##import Matsuoka_cl as OSC

##kwargs_mats={'numOsc':4,'h':1e-2,'tau':1e-2,'T':1e-1,'a':10.5,\
##        'b':20.5,'c':0.08,'aa':3}
##osc=OSC.Matsuoka(**kwargs_mats)


kwargs_hopf={'numOsc':4,'h':1e-2,'alpha':5.,'beta':50.,'mu':1.,'w_stance':sol[iter_num][0],\
             'w_swing':sol[iter_num][1],'b':10.,'F':300,'feedback':0,'gait':0}
osc=OSC.Hopf(**kwargs_hopf)

# =================================================================================================================================================
#                                       Creating muscles

PP=sol[iter_num][2]

servo_ids = {}
servo_ids["wrist.L"]      = setPositionServo(reference_object_name = "obj_wrist.L",      attached_object_name = "obj_forearm.L", P = PP)
servo_ids["wrist.R"]      = setPositionServo(reference_object_name = "obj_wrist.R",      attached_object_name = "obj_forearm.R", P = PP)
servo_ids["forearm.L"]    = setPositionServo(reference_object_name = "obj_forearm.L",    attached_object_name = "obj_upper_arm.L", P = PP)
servo_ids["forearm.R"]    = setPositionServo(reference_object_name = "obj_forearm.R",    attached_object_name = "obj_upper_arm.R", P = PP)

servo_ids["upper_arm.L"]  = setPositionServo(reference_object_name = "obj_upper_arm.L",  attached_object_name = "obj_shoulder.L", P = PP)
servo_ids["upper_arm.R"]  = setPositionServo(reference_object_name = "obj_upper_arm.R",  attached_object_name = "obj_shoulder.R", P = PP)
servo_ids["shin_lower.L"] = setPositionServo(reference_object_name = "obj_shin_lower.L", attached_object_name = "obj_shin.L", P = PP)
servo_ids["shin_lower.R"] = setPositionServo(reference_object_name = "obj_shin_lower.R", attached_object_name = "obj_shin.R", P = PP)

servo_ids["shin.L"]       = setPositionServo(reference_object_name = "obj_shin.L",       attached_object_name = "obj_thigh.L", P = PP)
servo_ids["shin.R"]       = setPositionServo(reference_object_name = "obj_shin.R",       attached_object_name = "obj_thigh.R", P = PP)
servo_ids["thigh.L"]       = setPositionServo(reference_object_name = "obj_thigh.L",     attached_object_name = "obj_hips", P = PP)
servo_ids["thigh.R"]       = setPositionServo(reference_object_name = "obj_thigh.R",     attached_object_name = "obj_hips", P = PP)



###########################################
# To determine speed, another way
speed_len=100
speed_array=np.zeros((speed_len,1))

# =================================================================================================================================================
#                                       Evolve function
def evolve():
    # Global variable definitions
    global inpp,outp,PP
    global joints,nJoint,numRec
    global f, iter_num, sol, length_sol, f_sol
    #global speed_len,speed_array

    ## Speed first way
##    lv=scn.objects['obj_head'].localLinearVelocity
##    speed_array=np.vstack((lv[1],speed_array[:-1]))
##    sp=float(sum(speed_array)/speed_len)

    ## Speed second way
    vel=velocity('obj_head',t_bl,1)

    #print("Step:", i_bl, "  Time:{0:.2f}".format(t_bl),'   Vel:{0:8.2f}  {1:8.2f}'.format(sp,vel))
    print("Step:", i_bl, "  Time:{0:.2f}".format(t_bl),'   Vel:{0:8.2f}'.format(vel))
    
    # ------------------------------------- Visual ------------------------------------------------------------------------------------------------
    #visual_array     = getVisual(camera_name = "Meye", max_dimensions = [256,256])
    #scipy.misc.imsave("test_"+('%05d' % (i_bl+1))+".png", visual_array)
    # ------------------------------------- Olfactory ---------------------------------------------------------------------------------------------
    olfactory_array  = getOlfactory(olfactory_object_name = "obj_nose", receptor_names = ["smell1", "plastic1"])
    # ------------------------------------- Taste -------------------------------------------------------------------------------------------------
    taste_array      = getTaste(    taste_object_name =     "obj_mouth", receptor_names = ["smell1", "plastic1"], distance_to_object = 1.0)
    # ------------------------------------- Vestibular --------------------------------------------------------------------------------------------
    vestibular_array = getVestibular(vestibular_object_name = "obj_head")
    # ------------------------------------- Sensory -----------------------------------------------------------------------------------------------
    # ------------------------------------- Proprioception ----------------------------------------------------------------------------------------
    #~ spindle_FLEX = getMuscleSpindle(control_id = muscle_ids["forearm.L_FLEX"])
    #~ spindle_EXT  = getMuscleSpindle(control_id = muscle_ids["forearm.L_EXT"])
    # ------------------------------------- Neural Simulation -------------------------------------------------------------------------------------
    

    # ------------------------------------- Muscle Activation -------------------------------------------------------------------------------------
    
    speed_ = 6.0 # Speed of the mouse (ang. freq of joint patterns)

    # Joint signals to be applied
    act_tmp         = 0.5 + 0.5*np.sin(speed_*t_bl)
    anti_act_tmp    = 1.0 - act_tmp
    act_tmp_p1      = 0.5 + 0.5*np.sin(speed_*t_bl - np.pi*0.5)
    anti_act_tmp_p1 = 1.0 - act_tmp_p1
    act_tmp_p2      = 0.5 + 0.5*np.sin(speed_*t_bl + np.pi*0.5)
    anti_act_tmp_p2 = 1.0 - act_tmp_p2


    osc.iterate(1)
    y=osc.output()
    for i in range(len(y)):
        if y[i]<=0.:
            y[i]=0.001
        elif y[i]>=1.:
            y[i]=0.999
    

##    joints=["wrist.L","wrist.R","forearm.L","forearm.R",\
##            "upper_arm.L","upper_arm.R","shin_lower.L","shin_lower.R",\
##            "shin.L","shin.R","thigh.L","thigh.R"]
            
    # Reference value of joints
    r=np.array([0.4,0.4,0.8*y[0],0.8*y[2],\
                y[0],y[2],0.8*y[3], 0.8*y[1],\
                0.5*y[3],0.5*y[1],0.5*y[3],0.5*y[1]])    
    
   # positions=np.array([getMuscleSpindle(control_id = servo_ids[joints[i]])[0] for i in range(len(joints))])    
    
    for i in range(nJoint):
        controlActivity(control_id = servo_ids[joints[i]], control_activity = r[i])
        # Apply the reference 


    # Get actual positions of joints after the control input
    positions=np.array([getMuscleSpindle(control_id = servo_ids[joints[i]])[0] for i in range(len(joints))])

    
    numRec=3
    inpp.append(r[numRec]) # Record reference
    outp.append(positions[numRec]) # record positions of servos

    if i_bl==250:
        data=np.hstack((np.array(inpp),np.array(outp),joints[numRec],PP,speed_))
        PickleIt(data,'inpOut') # Save the data to be plotted
        osc.plot()
        

    if np.mod(i_bl,300)==299:
        ff=f(vel)   # Get the value of the solutions using objective function
        print('iter_num',iter_num,'length of sol',length_sol)
        f_sol[iter_num]=ff # Store the value of the solution
        iter_num+=1 # Increase the iteration number to pass to successive solution
        PickleIt([sol,f_sol,iter_num,0],'save_cma') # Save solutions and function values
        bge.logic.restartGame() # restart game
