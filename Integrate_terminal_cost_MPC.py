import numpy as np
import casadi as ca
import A3_conf as conf
from train import get_critic, tf2np, np2tf


from casadi.casadi import SX, MX, DM
from casadi import fmax, Function, transpose
from OCP_SinglePendulum import OcpSinglePendulumWithNNCost

if __name__=="__main__":
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    N = conf.N       # horizon size
    dt = conf.dt        # time step

    w_u = conf.w_u    # weight for control input
    w_v = conf.w_v    #weight for velocity cost
    w_x = conf.w_x    #weight for position cost

    nq = conf.nq  #number of joint position
    nv = conf.nv  #number of joint velocity
    nu = conf.nu  #number of control

    lowerPositionLimit = conf.lowerPositionLimit     # min joint position
    upperPositionLimit = conf.upperPositionLimit      # max joint position
    
    lowerVelocityLimit = conf.lowerVelocityLimit            # min joint velocity
    upperVelocityLimit = conf.upperVelocityLimit             # max joint velocity

    lowerControlBound    = conf.lowerControlBound    # lower bound joint torque
    upperControlBound    = conf.upperControlBound       # upper bound joint torque

    x_des_final = np.array([0,0])  #final desired joint velocity
    
    n = nq + nv #state size
    m = nu  #control size



    is_plot = False

    data = []
    # X = np.linspace(-2.2, 2.0, 100)

    #Start MPC
    x0 = np.array([np.pi/2, -3]) # Start at fixed state
    #X = np.random.uniform(lowerPositionLimit, upperPositionLimit, size =(n_ics, n))
    ocp = OcpSinglePendulumWithNNCost(dt, w_u,w_x, w_v, lowerControlBound, upperControlBound, lowerPositionLimit,upperPositionLimit,lowerVelocityLimit,upperVelocityLimit)


    U = np.zeros((N,m)) #initial control input

    x = np.zeros((N,n))
    x_opt = []
    for i in range(N):
        x[0,:] = x0 #initial state
        sol = ocp.OCP_setup(x[i,:], N, x_des_final,U_guess=U) #solve ocp with initial guess state x and u
        sol = ocp.solve()
        x_opt.append(x[i,0])
        print('state x',sol.value(ocp.x[0,:]))
        print(sol.value(ocp.cost[0], [ocp.x==sol.value(ocp.x[0,:])]))
        u_opt = sol.value(ocp.u[0]) # get the first optimal control input
        print('u optimal', u_opt)
        u_res = u_opt
        if i < N-1:
            x[i+1,0] = x[i,0] + dt*x[i,1] #dynamics
            x[i+1,1] = x[i,1] + dt * (u_res + 9.81 * ca.sin(x[i,0])) # apply the first optimal control input to get next state
        U = np.pad(U[1:N], (0, 1), 'constant')
        print(i)
    print('x_opt size',np.array(x_opt))
    t = [i for i in range(N)]
    # plt.imshow()
    # print('x',x_opt)
    #print('u',u)
    #print('t',t)
        
    plt.plot(t, x_opt, 'xr',label='state x')
    # #plt.stairs(u,t)
    plt.xlabel("Time")
    plt.legend()
    plt.show()
        #costs = [sol.value(ocp.cost[0], [ocp.x==x_val]) for x_val in X[:,0]]
        #opt_cost_J = float(np.min(costs))
        #data.append({"x0":q0, "j_opt":opt_cost_J})
    if is_plot:
        plt.plot(X, costs)
        for i in range(N+1):
            plt.plot(sol.value(ocp.x[i]), sol.value(ocp.running_costs[i]), 
                            'xr', label='x_'+str(i))
        plt.legend()
        plt.show()
    
    # json.dump(data, open("valid.json", "w"))
    # df = pd.DataFrame(data)

    # data_test = json.load(open("test.json"))
    # df_test = pd.DataFrame(data_test)
    # print(df_test)
    # df.index = df["x0"]
    # df_test.index = df_test["x0"]

    # df_test.columns = ["x0", "y_val"]

    # df=df.join(df_test, lsuffix="_l", rsuffix="_r")

    # df.loc[:, "y_dif"] = df.loc[:, "y_val"] - df.loc[:, "j_opt"]
    # print(df)
    # print("error:      ", df["y_dif"].unique())


    # # load json file:
    
    # print(data)