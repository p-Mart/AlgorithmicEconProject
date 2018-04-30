import tensorflow as tf
import numpy as np
import sys, os, errno
import random
from tqdm import trange

def softmax(x):
    z = x - np.max(x)
    return np.exp(z) / np.sum(np.exp(z))

def replicator(G, x, y, delta=1.0):
    x, y = (softmax(x), softmax(y))

    A, B = G()

    delta_x = x*(A.dot(y) - x.T.dot(A).dot(y))
    delta_y = y*(x.T.dot(B).T - x.T.dot(B).dot(y))

    return delta_x, delta_y

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), (size, 5))

class DQN:
    def __init__(self, learning_rate, nb_actions, observation_shape, scope, h_size=128):
        with tf.variable_scope(scope):
            self.input = tf.placeholder(shape=[None]+list(observation_shape), dtype=tf.float32)

            # Architecture
            self.h_1 = tf.layers.dense(
                inputs=self.input,
                units=h_size,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.1)
            )

            self.h_2 = tf.layers.dense(
                inputs=self.h_1,
                units=h_size,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.1)
            )

            self.flatten = tf.layers.flatten(
                inputs = self.h_2
            )

            self.policy = tf.layers.dense(
                inputs=self.flatten,
                units=nb_actions,
                activation=tf.nn.softmax,
                use_bias=False,
                name="policy_out"
            )

            self.output =tf.layers.dense(
                inputs=self.flatten,
                units=nb_actions
            )

            self.predict = tf.argmax(self.output, axis=1)

            # Everything needed to calculate loss function
            self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, nb_actions, dtype=tf.float32)

            self.q = tf.reduce_sum(tf.multiply(self.output, self.actions_onehot), axis=1)

            self.loss = tf.losses.mean_squared_error(self.target_q, self.q)
            self.policy_loss = tf.reduce_sum(tf.multiply(-tf.log(self.policy), self.actions_onehot), axis=1)

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            self.update_q = self.optimizer.minimize(self.loss)

            # Gradient blocking on policy branch
            with tf.variable_scope('policy_out', reuse=True):
                w = tf.get_variable('kernel')

            self.update_policy = self.optimizer.minimize(self.policy_loss, var_list=[w])

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def updateTargetNetwork(op_holder, sess):
    for op in op_holder:
        sess.run(op)

def Stag():
    A = [
        [2, 0],
        [1, 1]
    ]

    B = [
        [2, 1],
        [0, 1]
    ]

    return np.array(A), np.array(B)

def PrisonersDilemma():
    A = [
        [-1, -3],
        [0, -2]
    ]

    B = [
        [-1, 0],
        [-3, -2]
    ]

    return np.array(A), np.array(B)

def Matching():
    A = [
        [1, -1],
        [-1, 1]
    ]

    B = [
        [-1, 1],
        [1, -1]
    ]

    return np.array(A), np.array(B)

def summaryWrite(name, value, summary_writer, step):
    new_summary = tf.Summary()
    new_summary_value = new_summary.value.add()
    new_summary_value.simple_value = value
    new_summary_value.tag = name

    summary_writer.add_summary(new_summary, (step + 1))
    summary_writer.flush()

def main(model_name, options):
    # Hyperparameters
    batch_size = 16
    update_freq = 10
    save_freq = 1000
    learning_rate = 0.003
    discount = 0.99
    replay_size = 1000
    nb_epochs = 10000
    nb_episodes = 25
    h_size = 4
    gamma = 0.4

    # Initialize 2 Player normal-form game
    A, B = Stag()

    # Env parameters
    nb_actions = 2
    observation_shape = (2,)

    # Initialize networks, tensorflow vars
    tf.reset_default_graph()

    # DQN for player 1
    model_1 = DQN(
        learning_rate=learning_rate,
        h_size=h_size,
        nb_actions=nb_actions,
        observation_shape=observation_shape,
        scope='model_1'
    )

    target_model_1 = DQN(
        learning_rate=learning_rate,
        h_size=h_size,
        nb_actions=nb_actions,
        observation_shape=observation_shape,
        scope='target_model_1'
    )

    # DQN for player 2
    model_2 = DQN(
        learning_rate=learning_rate,
        h_size=h_size,
        nb_actions=nb_actions,
        observation_shape=observation_shape,
        scope='model_2'
    )

    target_model_2 = DQN(
        learning_rate=learning_rate,
        h_size=h_size,
        nb_actions=nb_actions,
        observation_shape=observation_shape,
        scope='target_model_2'
    )

    # Tensorflow Stuff
    sess = tf.Session()

    summary_writer = tf.summary.FileWriter(logdir="./logs/{}".format(model_name),graph=sess.graph)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    trainables = tf.trainable_variables()

    target_ops_1 = update_target_graph('model_1', 'target_model_1')
    target_ops_2 = update_target_graph('model_2', 'target_model_2')

    experience_1 = ReplayBuffer(buffer_size=replay_size)
    experience_2 = ReplayBuffer(buffer_size=replay_size)

    sess.run(init)

    # Meta Strategy Initialization
    dummy = np.array([0, 1])
    meta_1 = softmax(np.random.uniform(size=(nb_actions,)))
    meta_2 = softmax(np.random.uniform(size=(nb_actions,)))
    #meta_1 = np.array([0.1, 0.9])
    #meta_2 = np.array([0.1, 0.9])

    p1_action_store = None
    p2_action_store = None

    p1_accumulated_reward = np.array([0.0, 0.0])
    p1_reward = 0. # For logging
    p2_accumulated_reward = np.array([0.0, 0.0])
    p2_reward = 0. # For logging

    total_steps = 0

    log_dir = "./data/{}/".format(model_name)
    # Weights will be loaded if weight file exists.
    if os.path.exists(log_dir) and "load" in options:
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)

    # Attempt to create log directory for this model.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    epoch = 0
    # Training loop, if enabled
    if "train" in options:
        episode = 0
        try:
            for epoch in trange(nb_epochs):

                p1_reward = 0.
                p2_reward = 0.
                p1_accumulated_reward = np.array([0.0, 0.0])
                p2_accumulated_reward = np.array([0.0, 0.0])

                # Player 1 Loop
                for episode in trange(nb_episodes):
                    # Sample from player 2's meta distribution
                    idx = np.random.choice(dummy, p=meta_2)

                    observation = meta_2
                    observation_next = meta_2

                    # Player 2 action
                    p2_action = None
                    if idx == 0:
                        p2_action = np.random.choice(dummy)
                    else:
                        p2_action = sess.run(model_2.predict, feed_dict={model_2.input:[meta_1]})[0]

                    idx = np.random.choice(dummy, p=meta_1)
                    p1_action = None
                    p2_policy = sess.run(model_2.policy,feed_dict={model_2.input:[meta_1]})
                    p1_policy = sess.run(model_1.policy,feed_dict={model_1.input:[p2_policy.flatten()]})

                    if idx == 0:
                        p1_action = np.random.choice(dummy)
                    else:
                        p1_action = sess.run(model_1.predict,feed_dict={model_1.input:[p2_policy.flatten()]})[0]

                    p1_action_store = p1_action

                    #reward = p1_policy.dot(A).dot(p2_policy.T)[0,0]
                    reward = A[p1_action][p2_action]
                    p1_accumulated_reward[idx] += reward
                    p1_reward += reward

                    done = True if (episode == nb_episodes - 1) else False

                    new_experience = np.array([p2_policy.flatten(), p1_action, reward, observation_next, done])
                    new_experience = new_experience.reshape((1, 5))
                    experience_1.add(new_experience)


                # Train Player 1
                train_batch = experience_1.sample(batch_size)

                s0_batch = np.stack(train_batch[:, 0], axis=0)
                s1_batch = np.stack(train_batch[:, 3], axis=0)

                model_q = sess.run(model_1.predict, feed_dict={model_1.input: s1_batch})
                target_model_q = sess.run(target_model_1.output, feed_dict={target_model_1.input: s1_batch})

                end_multiplier = 1 - train_batch[:, 4]
                double_q = target_model_q[range(batch_size), model_q]
                target_q = train_batch[:, 2] + (discount * double_q * end_multiplier)

                # Update model
                _ = sess.run([model_1.update_q, model_1.update_policy], feed_dict={
                    model_1.input: np.stack(s0_batch, axis=0),
                    model_1.target_q: target_q,
                    model_1.actions: train_batch[:, 1]
                })


                # Update meta strategy (Hedge Matching without exploration)
                meta_1 = softmax((gamma / 3.0) * p1_accumulated_reward)

                # Update target network 1
                if epoch % update_freq == 0:
                    updateTargetNetwork(target_ops_1, sess)

                # Player 2 Loop
                for episode in trange(nb_episodes):
                    # Sample from player 1's meta distribution
                    idx = np.random.choice(dummy, p=meta_1)

                    observation = meta_1
                    observation_next = meta_1

                    # Player 1 action
                    p1_action = None
                    if idx == 0:
                        p1_action = np.random.choice(dummy)
                    else:
                        p1_action = sess.run(model_1.predict, feed_dict={model_1.input: [meta_2]})[0]

                    idx = np.random.choice(dummy, p=meta_2)

                    p2_action = None
                    p1_policy = sess.run(model_1.policy,feed_dict={model_1.input:[meta_2]})
                    p2_policy = sess.run(model_2.policy,feed_dict={model_2.input:[p1_policy.flatten()]})

                    if idx == 0:
                        p2_action = np.random.choice(dummy)
                    else:
                        p2_action = sess.run(model_2.predict, feed_dict={model_2.input: [p1_policy.flatten()]})[0]

                    p2_action_store = p2_action

                    done = True if (episode == nb_episodes - 1) else False

                    #reward_random = np.array([[0.5, 0.5]]).dot(B).dot(p1_policy.T)[0,0]
                    #reward = p2_policy.dot(B).dot(p1_policy.T)[0,0]

                    reward = B[p1_action][p2_action]
                    p2_accumulated_reward[idx] += reward
                    p2_reward += reward

                    new_experience = np.array([p1_policy.flatten(), p2_action, reward, observation_next, done])
                    new_experience = new_experience.reshape((1, 5))
                    experience_2.add(new_experience)

                # Train Player 2
                train_batch = experience_2.sample(batch_size)

                s0_batch = np.stack(train_batch[:, 0], axis=0)
                s1_batch = np.stack(train_batch[:, 3], axis=0)

                model_q = sess.run(model_2.predict, feed_dict={model_2.input: s1_batch})
                target_model_q = sess.run(target_model_2.output, feed_dict={target_model_2.input: s1_batch})

                end_multiplier = 1 - train_batch[:, 4]
                double_q = target_model_q[range(batch_size), model_q]
                target_q = train_batch[:, 2] + (discount * double_q * end_multiplier)

                # Update model
                _ = sess.run([model_2.update_q, model_2.update_policy], feed_dict={
                    model_2.input: np.stack(s0_batch, axis=0),
                    model_2.target_q: target_q,
                    model_2.actions: train_batch[:, 1]
                })

                # Update meta strategy (Hedge Matching without exploration)
                meta_2 = softmax((gamma / 3.0) * p2_accumulated_reward)

                # Update target network
                if epoch % update_freq == 0:
                    updateTargetNetwork(target_ops_2, sess)

                # Log reward in tensorboard
                summaryWrite('player1/p1_action', p1_action_store, summary_writer, epoch+1)
                summaryWrite('player1/meta_1', meta_1[0], summary_writer, epoch+1)
                summaryWrite('player1/meta_2', meta_1[1], summary_writer, epoch+1)
                summaryWrite('player1/policy_1', p1_policy[0,0], summary_writer, epoch+1)
                summaryWrite('player1/policy_2', p1_policy[0,1], summary_writer, epoch+1)
                summaryWrite('player1/reward', p1_reward / nb_episodes, summary_writer, epoch+1)


                summaryWrite('player2/p2_action', p2_action_store, summary_writer, epoch+1)
                summaryWrite('player2/meta_1', meta_2[0], summary_writer, epoch+1)
                summaryWrite('player2/meta_2', meta_2[1], summary_writer, epoch+1)
                summaryWrite('player2/policy_1', p2_policy[0,0], summary_writer, epoch+1)
                summaryWrite('player2/policy_2', p2_policy[0,1], summary_writer, epoch+1)
                summaryWrite('player2/reward', p2_reward / nb_episodes, summary_writer, epoch+1)


                # Save model occasionally
                if epoch % save_freq == 0:
                    saver.save(sess, log_dir+'/model-'+str(epoch+1)+'.ckpt')

        except KeyboardInterrupt:
            print("Training interrupted manually.")

        # Save model
        saver.save(sess, log_dir + '/model-' + str(epoch+1) + '.ckpt')

        # Log all hyperparameter settings
        #logHyperparameters(model_name, **hyperparameters)

    # Test loop, if enabled
    '''
    if "test" in options:
        episode = 0
        try:
            for episode in trange(nb_episodes_test):
                observation = env.reset()
                observation = preprocess(observation, processed_resolution)
                done = False
                r_all = 0
                steps = 0

                while not done:
                    steps += 1
                    total_steps += 1

                    # Only the network is used to take an action
                    action = sess.run(model.predict, feed_dict={model.input:[observation]})[0]
                    print(action)
                    # Step through environment
                    env.render()
                    observation, reward, done, _ = env.step(action)
                    observation = preprocess(observation, processed_resolution)

                    # Store reward
                    r_all += reward

                    if done:
                        break

                # Log reward in tensorboard
                # TODO : Make this into a function pls
                episode_reward_summary = tf.Summary()
                reward_value = episode_reward_summary.value.add()
                reward_value.simple_value = r_all
                reward_value.tag = 'reward_test'
                summary_writer.add_summary(episode_reward_summary, episode)
                summary_writer.flush()

                steps_per_episode.append(steps)
                reward_list.append(r_all)
        except KeyboardInterrupt:
            print("Testing interrupted manually.")

    '''
    return


if __name__ == "__main__":

    model_name = None
    options = []

    if len(sys.argv) >= 3:
        model_name = sys.argv[1]

        mode = sys.argv[2]

        if mode != "train" and mode != "test":
            print("Usage: python main.py [model_name].h5 [train | test]")
            sys.exit()

        options = sys.argv[2:]

    else:
        print("Incorrect number of arguments.")
        print("Usage: python main.py [model_name].h5 [train | test]")
        sys.exit()

    main(model_name, options)