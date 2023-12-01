# Intrusion Detection System for IoT devices using Federated Learning

Clément Safon, Sarah Ramdani 

*Télécom SudParis, Palaiseau, France*

## Abstract 

> With IoT systems being on the rise the past few years, attacks on these devices have also been present. In this research project, we aimed to look at different ways to train models, with first a binary classification, then multi-class. 


## I. Introduction 

## Dataset presentation

The dataset on which this project is based is the UNSW-NB15 dataset. Before this dataset, two others were largely looked upon: KDDCUP99 and NSLKDD. However, these two datasets were outdated, and not representative of the [[1]](#1)

2,540,044 flows of traffic were opened for this dataset. 
Eleven types of attacks were categorized, as follows:
- Fuzzers
- Analysis 
- DoS 
- Exploits
- Generic
- Reconnaissance 
- Shellcode 
- Backdoors 
- Worms

It is important to notice that in this dataset, the distribution is not equal for every attack [[2]](#2), and quite different in some ways. This can largely affect the classifications (see results).
The features were classified into three groups : Basic, Content, and Time. '0' refers to, in the dataset, as a regular behavior, while '1' refers to a malicious behaviour. 

## References
<a id="1">[1]</a> 
Moustafa, Nour & Slay, Jill. ([2015](https://www.researchgate.net/publication/287330529_UNSW-NB15_a_comprehensive_data_set_for_network_intrusion_detection_systems_UNSW-NB15_network_data_set)). 
*UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set).*
10.1109/MilCIS.2015.7348942. 

<a id="2">[2]</a>
Zoghi, Zeinab & Serpen, Gursel. ([2021](https://arxiv.org/abs/2101.05067)).
*UNSW-NB15 Computer Security Dataset: Analysis through 
Visualization.*
Electrical Engineering & Computer Science, University of Toledo, Ohio, USA.

<a id="3">[3]</a>


<a id="4">[4]</a>
Zhang, Michael, et al. ([2020](https://arxiv.org/abs/2012.08565)).
"Personalized federated learning with first order model optimization." arXiv preprint arXiv:2012.08565 

