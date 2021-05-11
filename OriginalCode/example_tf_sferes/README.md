This is an example showing how to use sferes and tensorflow. 
It reproduces the "ballistic task" of the AURORA paper (GECCO'19)

To run it, you need to use the docker environment including tensorflow. 
It can be found in the [airl_env repository on the tensorflow branch](https://gitlab.doc.ic.ac.uk/AIRL/airl_env/tree/tensorflow)

When you are in the docker container, you can configure the experiment with:
`./waf configure --exp example_tf_sferes --kdtree /workspace/include`
and then compile it with:
`./waf --exp example_tf_sferes `

Finally, you can run it with: 
`build/exp/example_tf_sferes/example`


This experiment is composed of two parts: 
1) Some python code (in exp/example_tf_sferes/python) is used to generate a graph. 
2) This graph is loaded by sferes in cpp (in exp/example_tf_sferes/cpp/modifier_dim_red.hpp) and used in the QD experiment as described un the AURORA paper. 

