# Forest & Boost Prompts

Every prompt from this chat, word for word, in order.

1. how are you today

2. whats the simplest way to get mlflow into my uv project, thinking about updating bcancer.py, or should i be updating the app.py, thinking about updating our logreg with decision trees, can you guide me to learn along the way, ive heard of hyperparameters, can we run some decision trees with different hyperparameters, can you describe them to me when you use them, pretty sure mlflow does this, and captures everything me need, let me know what i missed

3. looks like logreg wins, can we try more realsies hyperparams for decision trees, also don't run it, i will run it, and then save the best model via joblib

4. (pasted MLflow UI output after running `bcancer.py`)

   ```text
   State:Active
                   
                   
                   
               
                   
                   
                   
               
                   
                   
                   
               
                   
                   
                       
                   
                   
                   
                   
               
                   
                   
                   
               
                   
                   
                   
               
                   
                   
                   
               
                   
                   
                   
               tree_class_weight-balanced2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_ccp_alpha-pruned2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=none_leaf=10_crit=entropy2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=none_leaf=10_crit=gini2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=none_leaf=5_crit=entropy2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=none_leaf=5_crit=gini2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=none_leaf=1_crit=entropy2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=none_leaf=1_crit=gini2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=8_leaf=10_crit=entropy2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=8_leaf=10_crit=gini2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=8_leaf=5_crit=entropy2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=8_leaf=5_crit=gini2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=8_leaf=1_crit=entropy2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=8_leaf=1_crit=gini2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=5_leaf=10_crit=entropy2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=5_leaf=10_crit=gini2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=5_leaf=5_crit=entropy2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=5_leaf=5_crit=gini2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=5_leaf=1_crit=entropy2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=5_leaf=1_crit=gini2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=3_leaf=10_crit=entropy2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=3_leaf=10_crit=gini2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=3_leaf=5_crit=entropy2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=3_leaf=5_crit=gini2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=3_leaf=1_crit=entropy2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree_depth=3_leaf=1_crit=gini2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runlogistic-regression2 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree-criterion-entropy16 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree-min_samples_split-1016 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree-min_samples_leaf-516 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree-max_depth-516 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree-max_depth-316 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree-unconstrained16 minutes ago
                   
                   
                   
                       
                   
               
               Pin runlogistic-regression16 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree-criterion-entropy33 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree-min_samples_split-1033 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree-min_samples_leaf-533 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree-max_depth-533 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree-max_depth-333 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree-unconstrained33 minutes ago
                   
                   
                   
                       
                   
               
               Pin runlogistic-regression33 minutes ago
                   
                   
                   
                       
                   
               
               Pin runtree-unconstrained34 minutes ago
                   
                   
                   
                       
                   
               
               Pin runlogistic-regression34 minutes ago
                   
                   
                   
                       
                   
               
               Pin run-245msbcancer.pymodel0.94736842105263150.96202531645569620.9743589743589743-247msbcancer.pymodel0.96491228070175440.9750.975-242msbcancer.pymodel0.96491228070175440.9756097560975610.9523809523809523-234msbcancer.pymodel0.94736842105263150.96296296296296290.9512195121951219-253msbcancer.pymodel0.94736842105263150.96296296296296290.9512195121951219-249msbcancer.pymodel0.96491228070175440.9750.975-241msbcancer.pymodel0.94736842105263150.96296296296296290.9512195121951219-236msbcancer.pymodel0.92982456140350880.950.95-245msbcancer.pymodel0.96491228070175440.9756097560975610.9523809523809523-227msbcancer.pymodel0.94736842105263150.96296296296296290.9512195121951219-254msbcancer.pymodel0.94736842105263150.96296296296296290.9512195121951219-242msbcancer.pymodel0.96491228070175440.9750.975-252msbcancer.pymodel0.94736842105263150.96296296296296290.9512195121951219-259msbcancer.pymodel0.92982456140350880.950.95-233msbcancer.pymodel0.96491228070175440.9756097560975610.9523809523809523-253msbcancer.pymodel0.94736842105263150.96296296296296290.9512195121951219-244msbcancer.pymodel0.96491228070175440.9756097560975610.9523809523809523-270msbcancer.pymodel0.96491228070175440.9750.975-247msbcancer.pymodel0.94736842105263150.96296296296296290.9512195121951219-248msbcancer.pymodel0.94736842105263150.96202531645569620.9743589743589743-240msbcancer.pymodel0.96491228070175440.9756097560975610.9523809523809523-247msbcancer.pymodel0.94736842105263150.96296296296296290.9512195121951219-253msbcancer.pymodel0.96491228070175440.9756097560975610.9523809523809523-253msbcancer.pymodel0.94736842105263150.96296296296296290.9512195121951219-241msbcancer.pymodel0.96491228070175440.9756097560975610.9523809523809523-238msbcancer.pymodel0.94736842105263150.96296296296296290.9512195121951219-4.3sbcancer.pymodel0.98245614035087710.98765432098765430.9756097560
   ```

5. I think I would like to try randomforestclassifier as another experiment, does it have hyperparameters, that can be tweaked and ran in multiple experiments, make sure to preserve all the existing experiments, dont run the bcancer.py, i will run it when you have it ready

6. logreg still wins, can we tweak a few more params and see if we can overfit a forest, just kidding

7. how about gradient boost experiments, and tell me why we dont combine all experiments together, is it because hyperparameters are diff per model

8. give me a forest-and-boost.md file with my prompts word for word

9. sure, include the whole chat
