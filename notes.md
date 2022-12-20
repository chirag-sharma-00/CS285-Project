# Project Notes

### Dec. 19, 2022 
#### Here's a list of changes made to the code since the first draft of the report:
- Fixed a **target update frequency bug**. Target networks were updated agent_num times more frequent. Now the update frequency is as specified by `critic_target_update_frequency`
- Fixed a **target network advice sourse bug**. Now target networks get advice from other target networks instead of from critic networks when training. 
- Changed advice behavior. We are now using advice as **context** to the observations, meaning that every where we have observatin we have advice with probability $1 - \epsilon$. This includes when actor calls critic and when calculating target q values using the target critic networks. 
- Added **self_advice** option for better symmetry. When self advicing, the advice is generated the network under-training it self. 

#### Todo:
- Investigate why our SAC implementation is not performing as well as in the papers here: https://arxiv.org/pdf/1802.09477.pdf and here https://arxiv.org/pdf/1801.01290.pdf
- Decide on where random peer should be chosen: inside critic forward call vs. inside agent train function. The latter ensures that the same peer is used for all advice calls in an training iteration. 
- Determin evaluation metrics for publishable results. Conduct larger scale experiments to characterize variance.
- Clearn up repo. Cluster .sh, .png files
- Update report with the changes mentioned above. 

#### Wishlist
- Figure out video generation
- Think about ways to visualize the advices being generated