import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt

# --- Подключение ---
client_gui = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# --- Загрузка модели ---
robot_gui = p.loadURDF(r"c:\\Users\\devya\\Desktop\\nir22\\triple_pendulum.urdf", useFixedBase=True)

# --- Револьвентные суставы ---
revolute_joints = [1, 3, 5]
eef_link = 6
print("Revolute joints:", revolute_joints)

# --- Отключаем встроенные моторы ---
for j in revolute_joints:
    p.setJointMotorControl2(robot_gui, j, controlMode=p.VELOCITY_CONTROL, force=0)

# --- Дополнительный клиент для "своей" модели ---
client_direct = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_direct)
robot_direct = p.loadURDF(r"c:\\Users\\devya\\Desktop\\nir22\\triple_pendulum.urdf",
                          useFixedBase=True,
                          physicsClientId=client_direct)
for j in revolute_joints:
    p.setJointMotorControl2(robot_direct, j, controlMode=p.VELOCITY_CONTROL,
                            force=0, physicsClientId=client_direct)

# --- Настройки симуляции ---
p.setTimeStep(1.0 / 240.0)
steps = 3000
dt = 1.0 / 240.0

# --- Целевая траектория (круг в плоскости XZ) ---
R = 0.5
center = np.array([0.0, 0.0, 1.5])
omega = 0.5

def desired_trajectory(t):
    xd = center + np.array([R*np.cos(omega*t), 0.0, R*np.sin(omega*t)])
    return xd

# --- ПИД-параметры ---
Kp, Kd = 50, 5
damping_coeff = 1.0

# --- Логирование ---
eef_traj, xd_traj = [], []
eef_my_traj = []

# --- Начальные условия для своей модели ---
q_my = np.zeros(3)
qd_my = np.zeros(3)

def fk_using_direct(q_vec):
    """Прямая кинематика для своей модели через DIRECT-клиент."""
    for j_idx, qv in zip(revolute_joints, q_vec):
        p.resetJointState(robot_direct, j_idx, qv, physicsClientId=client_direct)
    pos = p.getLinkState(robot_direct, eef_link,
                         computeForwardKinematics=True,
                         physicsClientId=client_direct)[0]
    return np.array(pos)

# --- Основной цикл ---
for step in range(steps):
    t = step * dt

    # --- Целевая точка ---
    xd = desired_trajectory(t)

    # --- IK через PyBullet ---
    qd_full = p.calculateInverseKinematics(robot_gui, eef_link, xd)
    qd = [qd_full[idx] for idx, j in enumerate(revolute_joints)]

    # --- Считываем состояния PyBullet ---
    q = [p.getJointState(robot_gui, j)[0] for j in revolute_joints]
    qdot = [p.getJointState(robot_gui, j)[1] for j in revolute_joints]

    # --- Управляющее ускорение ---
    qdd = [-Kp*(qi - qdi) - Kd*(dqi - 0.0)
           for qi, dqi, qdi in zip(q, qdot, qd)]

    # --- PyBullet inverse dynamics ---
    tau = p.calculateInverseDynamics(robot_gui, q, qdot, qdd)
    for j, t_j in zip(revolute_joints, tau):
        p.setJointMotorControl2(robot_gui, j, controlMode=p.TORQUE_CONTROL, force=t_j)

    # --- Собственная модель (semi-implicit Euler + демпфирование) ---
    qdd_my = np.array([-Kp*(qi - qdi) - Kd*(dqi - 0.0)
                       for qi, dqi, qdi in zip(q_my, qd_my, qd)])
    qdd_my -= damping_coeff * qd_my
    qd_my = qd_my + qdd_my * dt
    q_my = q_my + qd_my * dt

    # --- Позиции эффектора ---
    eef_pos = np.array(p.getLinkState(robot_gui, eef_link)[0])
    eef_traj.append(eef_pos)
    xd_traj.append(xd)

    eef_my_traj.append(fk_using_direct(q_my))

    p.stepSimulation()

print("Симуляция завершена")

# --- Перевод в numpy ---
eef_traj = np.array(eef_traj)
xd_traj = np.array(xd_traj)
eef_my_traj = np.array(eef_my_traj)
time_axis = np.linspace(0, steps*dt, steps)

# --- Метрики ---
error = eef_traj - xd_traj
L2 = np.sqrt(np.mean(np.sum(error**2, axis=1)))
Linf = np.max(np.abs(error))
print(f"Метрики PyBullet vs цель:")
print(f"L2-норма ошибки = {L2:.4f} м")
print(f"L∞-норма ошибки = {Linf:.4f} м")

# --- Графики ---
plt.figure(figsize=(9,6))
plt.plot(time_axis, xd_traj[:,0], '--', label="Цель X")
plt.plot(time_axis, eef_traj[:,0], label="PyBullet X")

plt.plot(time_axis, xd_traj[:,2], '--', label="Цель Y")
plt.plot(time_axis, eef_traj[:,2], label="PyBullet Y")

plt.xlabel("Время, c")
plt.ylabel("Координаты (м)")
plt.legend()
plt.title("PyBullet")
plt.grid()
plt.show()

plt.figure(figsize=(9,6))
plt.plot(time_axis, xd_traj[:,0], '--', label="Цель X")

plt.plot(time_axis, eef_my_traj[:,0], label="MyModel X")
plt.plot(time_axis, xd_traj[:,2], '--', label="Цель Y")

plt.plot(time_axis, eef_my_traj[:,2], label="MyModel Y")
plt.xlabel("Время, c")
plt.ylabel("Координаты (м)")
plt.legend()
plt.title("Собственная модель")
plt.grid()
plt.show()

# --- Отключение ---
p.disconnect(client_direct)

