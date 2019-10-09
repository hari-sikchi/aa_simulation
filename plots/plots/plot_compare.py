
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



dt_fixed = np.load("array_data_dt_0.02_baseline.npz")
dt_input = np.load("array_data_dt_train_baseline.npz")
lstm = np.load("array_data_lstm_baseline.npz")


print(dt_fixed['dt'].shape)
print(lstm['rewards'].shape)



plt.figure()
# Data
df=pd.DataFrame({'dt': dt_fixed['dt'].reshape(-1), 'dt_0.02': dt_fixed['rewards'].reshape(-1), 'dt_train': dt_input['rewards'], 'lstm_randomized': lstm['rewards'] }) 
# multiple line plot
plt.plot( 'dt', 'dt_0.02',  data=df, marker='', color='r', linewidth=2, linestyle='dashed')
plt.plot( 'dt', 'dt_train',  data=df, marker='', color='g', linewidth=2, linestyle='dashed')
plt.plot( 'dt', 'lstm_randomized', data=df, marker='', color='b', linewidth=2, linestyle='dashed')
plt.legend()
plt.ylabel('average return', fontsize=16)
plt.xlabel('dt', fontsize=16)
plt.savefig("compare_rewards.png")


plt.figure()

# Data
df=pd.DataFrame({'dt': dt_fixed['dt'].reshape(-1), 'dt_0.02': dt_fixed['vel_err'].reshape(-1), 'dt_train': dt_input['vel_err'], 'lstm_randomized': lstm['vel_err'] }) 
# multiple line plot
plt.plot( 'dt', 'dt_0.02',  data=df, marker='', color='r', linewidth=2, linestyle='dashed')
plt.plot( 'dt', 'dt_train',  data=df, marker='', color='g', linewidth=2, linestyle='dashed')
plt.plot( 'dt', 'lstm_randomized', data=df, marker='', color='b', linewidth=2, linestyle='dashed')
plt.legend()
plt.ylabel('average abs-vel error', fontsize=16)
plt.xlabel('dt', fontsize=16)
plt.savefig("compare_abs_vel_err.png")


plt.figure()

# Data
df=pd.DataFrame({'dt': dt_fixed['dt'].reshape(-1), 'dt_0.02': dt_fixed['steer'].reshape(-1), 'dt_train': dt_input['steer'], 'lstm_randomized': lstm['steer'] }) 
# multiple line plot
plt.plot( 'dt', 'dt_0.02',  data=df, marker='', color='r', linewidth=2, linestyle='dashed')
plt.plot( 'dt', 'dt_train',  data=df, marker='', color='g', linewidth=2, linestyle='dashed')
plt.plot( 'dt', 'lstm_randomized', data=df, marker='', color='b', linewidth=2, linestyle='dashed')
plt.legend()
plt.ylabel('average std_steer_dev', fontsize=16)
plt.xlabel('dt', fontsize=16)
plt.savefig("compare_std_steer.png")


# plt.figure(figsize=(12, 4))
# plt.subplot(121)  # Left plot will show performance vs number of iterations
# plt.plot(x, mu_r, label=str(pop_size))
# plt.ylabel('average return', fontsize=16)
# plt.xlabel('num. iterations', fontsize=16)




# for pop_size, values in data.items():
#   mu_r = np.array(values)[:, 1]  # Use the performance of the best point
#   x = np.arange(len(mu_r)) + 1
#   plt.plot(x, mu_r, label=str(pop_size))
#   plt.ylabel('average return', fontsize=16)
#   plt.xlabel('num. iterations', fontsize=16)

# plt.subplot(122)  # Right plot will show performance vs number of points evaluated
# for pop_size, values in data.items():
#   mu_r = np.array(values)[:, 1]  # Use the performance of the best point
#   x = pop_size * (np.arange(len(mu_r)) + 1)
#   plt.plot(x, mu_r, label=str(pop_size))
#   plt.ylabel('average return', fontsize=16)
#   plt.xlabel('num. points evaluated', fontsize=16)

# plt.legend()
# plt.tight_layout()
# plt.savefig('cmaes_pop_size.png')
# plt.show()