# Project Notes

### Last update: 2022-12-24
#### Here's a list of changes made to the code since the first draft of the report:
- Fixed a **target update frequency bug**. Target networks were updated agent_num times more frequent. Now the update frequency is as specified by `critic_target_update_frequency`
- Fixed a **target network advice sourse bug**. Now target networks get advice from other target networks instead of from critic networks when training. 
- Changed advice behavior. We are now using advice as **context** to the observations, meaning that every where we have observatin we have advice with probability $1 - \epsilon$. This includes when actor calls critic and when calculating target q values using the target critic networks. 
- Added **self_advice** option for better symmetry. When self advicing, the advice is generated the network under-training it self. 

#### Todo:
- Change location of choosing random peer. Choose random peer inside agent train function. 
- Find a way to fix the target network. It's not supposed to change until we soft-update it. Right now, every time a peer gets advice from a target network, gradients are backpropagated through the target network and it changes.
- Find a way to add **positional encoding** in the advice which help the critic identify which agent the advice comes from. 
- Investigate multi-task RL/ML structures in the critic. It needs to estimate q-values (task 1) and give advice (task 2).
- **Investigate why our SAC implementation is not performing as well as in the papers here: https://arxiv.org/pdf/1802.09477.pdf and here https://arxiv.org/pdf/1801.01290.pdf**
- Conduct larger scale experiments to characterize variance. Most existing publications repeat the same curve **10 times** and calculate the mean and variance. We are going to repeat 6 times for now. 
- **Clearn up** repo. Cluster .sh, .png files
- Update report with the changes. 

#### Wishlist
- Figure out video generation
- Think about ways to visualize the advices being generated