Search.setIndex({docnames:["get-started/conda-installation","get-started/development-installation","get-started/index","index","modules/anvil/anvil","modules/anvil/anvil.benchmark_config","modules/anvil/anvil.scripts","modules/anvil/anvil.tests","modules/anvil/modules","sphinx-docs"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["get-started/conda-installation.rst","get-started/development-installation.rst","get-started/index.rst","index.rst","modules/anvil/anvil.rst","modules/anvil/anvil.benchmark_config.rst","modules/anvil/anvil.scripts.rst","modules/anvil/anvil.tests.rst","modules/anvil/modules.rst","sphinx-docs.rst"],objects:{"":{anvil:[4,0,0,"-"]},"anvil.benchmarks":{eigvals_from_sample:[4,1,1,""],fourier_transform:[4,1,1,""],free_scalar_theory:[4,1,1,""],free_theory_from_training:[4,1,1,""],plot_kinetic_eigenvalues:[4,1,1,""],table_kinetic_eigenvalues:[4,1,1,""],table_real_space_variance:[4,1,1,""]},"anvil.checkpoint":{Checkpoint:[4,2,1,""],InvalidCheckpointError:[4,4,1,""],InvalidTrainingOutputError:[4,4,1,""],TrainingOutput:[4,2,1,""],TrainingRuncardNotFound:[4,4,1,""],current_loss:[4,1,1,""],loaded_checkpoint:[4,1,1,""],loaded_model:[4,1,1,""],loaded_optimizer:[4,1,1,""],loaded_scheduler:[4,1,1,""],train_range:[4,1,1,""]},"anvil.checkpoint.Checkpoint":{load:[4,3,1,""]},"anvil.checkpoint.TrainingOutput":{as_input:[4,3,1,""],final_checkpoint:[4,3,1,""],get_config:[4,3,1,""]},"anvil.checks":{check_trained_with_free_theory:[4,1,1,""]},"anvil.config":{ConfigParser:[4,2,1,""]},"anvil.config.ConfigParser":{parse_bootstrap_sample_size:[4,3,1,""],parse_checkpoints:[4,3,1,""],parse_couplings:[4,3,1,""],parse_cp_id:[4,3,1,""],parse_cp_ids:[4,3,1,""],parse_epochs:[4,3,1,""],parse_lattice_dimension:[4,3,1,""],parse_lattice_length:[4,3,1,""],parse_n_batch:[4,3,1,""],parse_optimizer:[4,3,1,""],parse_optimizer_params:[4,3,1,""],parse_parameterisation:[4,3,1,""],parse_sample_interval:[4,3,1,""],parse_sample_size:[4,3,1,""],parse_save_interval:[4,3,1,""],parse_scheduler:[4,3,1,""],parse_scheduler_params:[4,3,1,""],parse_sigma:[4,3,1,""],parse_thermalization:[4,3,1,""],parse_training_output:[4,3,1,""],parse_training_outputs:[4,3,1,""],parse_window:[4,3,1,""],parse_windows:[4,3,1,""],produce_base_dist:[4,3,1,""],produce_bootstrap_seed:[4,3,1,""],produce_checkpoint:[4,3,1,""],produce_geometry:[4,3,1,""],produce_lattice_size:[4,3,1,""],produce_model_action:[4,3,1,""],produce_size_half:[4,3,1,""],produce_target_dist:[4,3,1,""],produce_training_context:[4,3,1,""],produce_training_geometry:[4,3,1,""],produce_use_multiprocessing:[4,3,1,""]},"anvil.core":{FullyConnectedNeuralNetwork:[4,2,1,""],Sequential:[4,2,1,""],model_to_load:[4,1,1,""]},"anvil.core.FullyConnectedNeuralNetwork":{forward:[4,3,1,""],training:[4,5,1,""]},"anvil.core.Sequential":{forward:[4,3,1,""],training:[4,5,1,""]},"anvil.distributions":{Gaussian:[4,2,1,""],PhiFourScalar:[4,2,1,""],gaussian:[4,1,1,""],phi_four:[4,1,1,""]},"anvil.distributions.Gaussian":{log_density:[4,3,1,""]},"anvil.distributions.PhiFourScalar":{action:[4,3,1,""],from_albergo2019:[4,3,1,""],from_bosetti2015:[4,3,1,""],from_nicoli2020:[4,3,1,""],from_standard:[4,3,1,""],log_density:[4,3,1,""]},"anvil.free_scalar":{FreeScalarEigenmodes:[4,2,1,""]},"anvil.free_scalar.FreeScalarEigenmodes":{gen_complex_normal:[4,3,1,""],gen_eigenmodes:[4,3,1,""],gen_real_space_fields:[4,3,1,""]},"anvil.geometry":{Geometry2D:[4,2,1,""],ShiftsMismatchError:[4,4,1,""]},"anvil.geometry.Geometry2D":{get_shift:[4,3,1,""],two_point_iterator:[4,3,1,""]},"anvil.layers":{AdditiveLayer:[4,2,1,""],AffineLayer:[4,2,1,""],BatchNormLayer:[4,2,1,""],CouplingLayer:[4,2,1,""],GlobalAffineLayer:[4,2,1,""],GlobalRescaling:[4,2,1,""],RationalQuadraticSplineLayer:[4,2,1,""]},"anvil.layers.AdditiveLayer":{forward:[4,3,1,""],training:[4,5,1,""]},"anvil.layers.AffineLayer":{forward:[4,3,1,""],training:[4,5,1,""]},"anvil.layers.BatchNormLayer":{forward:[4,3,1,""],training:[4,5,1,""]},"anvil.layers.CouplingLayer":{_active_ind:[4,5,1,""],_join_func:[4,5,1,""],_passive_ind:[4,5,1,""],training:[4,5,1,""]},"anvil.layers.GlobalAffineLayer":{forward:[4,3,1,""],training:[4,5,1,""]},"anvil.layers.GlobalRescaling":{forward:[4,3,1,""],training:[4,5,1,""]},"anvil.layers.RationalQuadraticSplineLayer":{forward:[4,3,1,""],training:[4,5,1,""]},"anvil.models":{affine_spline:[4,1,1,""],coupling_pair:[4,1,1,""],nice:[4,1,1,""],rational_quadratic_spline:[4,1,1,""],real_nvp:[4,1,1,""],spline_affine:[4,1,1,""]},"anvil.observables":{abs_magnetization_squared:[4,1,1,""],autocorrelation:[4,1,1,""],correlation_length_from_fit:[4,1,1,""],cosh_shift:[4,1,1,""],effective_pole_mass:[4,1,1,""],fit_zero_momentum_correlator:[4,1,1,""],inverse_pole_mass:[4,1,1,""],ising_energy:[4,1,1,""],low_momentum_correlation_length:[4,1,1,""],magnetic_susceptibility:[4,1,1,""],magnetization:[4,1,1,""],magnetization_autocorr:[4,1,1,""],magnetization_integrated_autocorr:[4,1,1,""],magnetization_optimal_window:[4,1,1,""],magnetization_series:[4,1,1,""],optimal_window:[4,1,1,""],second_moment_correlation_length:[4,1,1,""],susceptibility:[4,1,1,""],two_point_connected_correlator:[4,1,1,""],two_point_correlator:[4,1,1,""],zero_momentum_correlator:[4,1,1,""]},"anvil.plot":{example_configs:[4,1,1,""],field_component:[4,1,1,""],field_components:[4,1,1,""],plot_bootstrap_effective_pole_mass:[4,1,1,""],plot_bootstrap_ising_energy:[4,1,1,""],plot_bootstrap_multiple_numbers:[4,1,1,""],plot_bootstrap_single_number:[4,1,1,""],plot_bootstrap_susceptibility:[4,1,1,""],plot_bootstrap_two_point:[4,1,1,""],plot_bootstrap_zero_momentum_2pf:[4,1,1,""],plot_effective_pole_mass:[4,1,1,""],plot_example_configs:[4,1,1,""],plot_field_components:[4,1,1,""],plot_magnetization_autocorr:[4,1,1,""],plot_magnetization_integrated_autocorr:[4,1,1,""],plot_magnetization_series:[4,1,1,""],plot_two_point_correlator:[4,1,1,""],plot_zero_momentum_correlator:[4,1,1,""]},"anvil.sample":{LogRatioNanError:[4,4,1,""],acceptance:[4,1,1,""],calc_tau_chain:[4,1,1,""],configs:[4,1,1,""],gen_candidates:[4,1,1,""],metropolis_hastings:[4,1,1,""],metropolis_test:[4,1,1,""],random:[4,1,1,""],tau_chain:[4,1,1,""]},"anvil.scripts":{anvil_benchmark:[6,0,0,"-"],anvil_sample:[6,0,0,"-"],anvil_train:[6,0,0,"-"]},"anvil.scripts.anvil_benchmark":{BenchmarkSampleApp:[6,2,1,""],BenchmarkTrainApp:[6,2,1,""],BenchmarkTrainConfig:[6,2,1,""],main:[6,1,1,""]},"anvil.scripts.anvil_benchmark.BenchmarkSampleApp":{init_logging:[6,3,1,""]},"anvil.scripts.anvil_benchmark.BenchmarkTrainApp":{config_class:[6,5,1,""]},"anvil.scripts.anvil_benchmark.BenchmarkTrainConfig":{from_yaml:[6,3,1,""]},"anvil.scripts.anvil_sample":{SampleApp:[6,2,1,""],main:[6,1,1,""]},"anvil.scripts.anvil_sample.SampleApp":{config_class:[6,5,1,""]},"anvil.scripts.anvil_train":{TrainApp:[6,2,1,""],TrainConfig:[6,2,1,""],TrainEnv:[6,2,1,""],TrainError:[6,4,1,""],main:[6,1,1,""]},"anvil.scripts.anvil_train.TrainApp":{add_positional_arguments:[6,3,1,""],argparser:[6,3,1,""],config_class:[6,5,1,""],environment_class:[6,5,1,""],get_commandline_arguments:[6,3,1,""],run:[6,3,1,""]},"anvil.scripts.anvil_train.TrainConfig":{from_yaml:[6,3,1,""]},"anvil.scripts.anvil_train.TrainEnv":{init_output:[6,3,1,""]},"anvil.table":{table_autocorrelation:[4,1,1,""],table_correlation_length:[4,1,1,""],table_effective_pole_mass:[4,1,1,""],table_fit:[4,1,1,""],table_magnetization:[4,1,1,""],table_two_point_correlator:[4,1,1,""],table_two_point_scalars:[4,1,1,""],table_zero_momentum_correlator:[4,1,1,""]},"anvil.tests":{test_benchmark:[7,0,0,"-"],test_distributions:[7,0,0,"-"],test_geometry:[7,0,0,"-"]},"anvil.tests.test_benchmark":{test_benchmark_runs:[7,1,1,""]},"anvil.tests.test_distributions":{test_normal_distribution:[7,1,1,""]},"anvil.tests.test_geometry":{test_checkerboard:[7,1,1,""],test_indexing:[7,1,1,""],test_splitcart:[7,1,1,""],test_splitlexi:[7,1,1,""]},"anvil.train":{reverse_kl:[4,1,1,""],save_checkpoint:[4,1,1,""],train:[4,1,1,""],training_update:[4,1,1,""]},"anvil.utils":{Multiprocessing:[4,2,1,""],bootstrap_sample:[4,1,1,""],get_num_parameters:[4,1,1,""],handler:[4,1,1,""]},"anvil.utils.Multiprocessing":{target:[4,3,1,""]},anvil:{benchmark_config:[5,0,0,"-"],benchmarks:[4,0,0,"-"],checkpoint:[4,0,0,"-"],checks:[4,0,0,"-"],config:[4,0,0,"-"],core:[4,0,0,"-"],distributions:[4,0,0,"-"],free_scalar:[4,0,0,"-"],geometry:[4,0,0,"-"],layers:[4,0,0,"-"],models:[4,0,0,"-"],observables:[4,0,0,"-"],plot:[4,0,0,"-"],sample:[4,0,0,"-"],scripts:[6,0,0,"-"],table:[4,0,0,"-"],tests:[7,0,0,"-"],train:[4,0,0,"-"],utils:[4,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","exception","Python exception"],"5":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:exception","5":"py:attribute"},terms:{"0306017v4":4,"06413":4,"1000":4,"2003":4,"2697":4,"2s_k":4,"2x2":4,"2xlength":4,"3499":4,"3838":4,"3940":4,"9087":4,"9730":4,"case":4,"class":[4,6],"default":4,"final":4,"float":4,"function":4,"int":4,"new":[0,4],"return":4,"static":4,"true":4,"try":4,"while":4,For:4,Ising:4,The:[0,4,9],Then:[0,1],Useful:4,_active_ind:4,_join_func:4,_metropolis_hast:4,_normalising_flow:4,_passive_ind:4,_plot_example_config:4,_plot_field_compon:4,_sample_runcard_path:6,_variablefunctionsclass:4,_varianc:4,about:4,abov:4,abs:4,abs_magnetization_squar:4,accept:4,access:4,accord:4,achiev:0,across:4,act:4,action:4,activ:[0,4],add:6,add_positional_argu:6,addit:4,additivelay:4,affin:4,affine_splin:4,affinelay:4,after:4,afterward:4,against:6,algorith:4,alia:6,all:[0,4],allow:4,alpha:4,also:4,altern:4,although:4,anaconda:0,ani:[1,3,4,6],anvil:[0,1,2],anvil_benchmark:[4,8],anvil_sampl:[4,8],anvil_train:[4,8],api:8,app:6,append:4,appli:4,appropri:4,approxim:4,arang:4,architectur:4,arg:[4,6],argpars:6,argument:4,arrai:4,arxiv:4,as_input:4,ascend:4,associ:4,assum:4,autocorrel:4,automat:[1,4],automatic_windowing_funct:4,averag:4,axi:4,back:3,base:[4,6],base_dist:4,base_neg:4,batch:4,batchnormlay:4,bayesiain:4,becom:4,been:4,befor:4,below:4,benchmark:[6,7,8],benchmark_config:[4,8],benchmarksampleapp:6,benchmarktrainapp:6,benchmarktrainconfig:6,beta:4,between:4,bia:4,bias:4,bin:4,block:4,bool:4,boostrap:4,bootstrap:4,bootstrap_sampl:4,bootstrap_sample_s:4,bootstrap_se:4,both:4,bottom:4,box:4,build:3,built:[4,9],c_ise:4,c_quadrat:4,c_quartic:4,calc_tau_chain:4,calcul:4,call:4,callabl:4,can:[0,1,3,4,9],cannot:4,care:4,carlo:4,cartesian:4,chain:4,chang:[1,4],check:[6,8],check_trained_with_free_theori:4,checkerboard:4,checkpoint:8,circular:4,classmethod:[4,6],claus:4,clone:1,cmdline:6,code:[0,1,2,9],coeffici:4,column:4,com:4,compar:4,compil:4,complex:4,compon:4,composit:4,comput:4,concaten:4,conda:[1,2,3,9],config:[6,8],config_class:6,config_yml:6,configpars:[4,6],configur:4,congratul:0,consid:4,consider:4,constant:4,construct:4,constructor:4,contain:[4,6],content:8,context:4,contribut:4,convent:4,convert:4,coordin:4,copi:4,core:8,corner:4,correct:4,correctli:4,correl:4,correlation_length_from_fit:4,correspond:4,cosh:4,cosh_shift:4,cost:4,could:4,coupl:4,coupling_lay:4,coupling_pair:4,couplinglay:4,cp_id:4,creat:0,cumul:4,current:4,current_log_ratio:4,current_loss:4,custom:6,d_0:4,d_k:4,data:4,decorrel:4,default_figure_format:6,defe:4,defin:4,definit:4,delta:4,densiti:4,depend:[0,4],deriv:4,det:4,detail:4,determin:4,develop:[2,3],deviat:4,diagonos:6,dict:4,dictat:4,dictionari:4,differ:4,dim:4,dimens:4,dimension:4,direct:4,directori:4,discard:4,distribut:[7,8],diverg:4,divid:4,doc:[3,4,9],docstr:4,document:4,doe:4,domain:4,don:4,done:4,down:4,download:1,drawn:4,due:4,each:4,effect:4,effective_pole_mass:4,effort:4,eigenmod:4,eigenvalu:4,eigvals_from_sampl:4,either:[0,4],element:4,end:4,energi:4,enough:4,ensur:4,entri:4,environ:[0,1,4,6,9],environment_class:6,epoch:4,eps:4,equal:4,equival:4,error:4,errorbar:4,estim:4,even:4,even_sit:4,everi:4,exampl:[4,7],example_config:4,except:[4,6],execut:4,exercis:7,exit:4,exp:4,expect:7,explan:4,extend:4,extens:4,extra:4,extract:4,factor:4,fals:4,featur:4,field:4,field_compon:4,figur:4,fill:6,final_checkpoint:4,finit:4,first:4,fit_zero_momentum_correl:4,flag:4,flow:4,follow:[1,4],form:4,format:6,former:4,formula:4,forward:4,found:9,four:4,fourier:4,fourier_transform:4,frac:4,fraction:4,frame:4,free:[4,6],free_scalar:8,free_scalar_theori:4,free_theory_from_train:4,free_theory_from_training_:4,freescalareigenmod:4,from:[4,6,9],from_albergo2019:4,from_bosetti2015:4,from_nicoli2020:4,from_standard:4,from_yaml:6,fullyconnectedneuralnetwork:4,func:4,futur:7,g_1:4,g_2:4,g_n:4,gaussian:4,gen_candid:4,gen_complex_norm:4,gen_eigenmod:4,gen_real_space_field:4,gener:[4,7],geom:4,geometri:[7,8],geometry2d:4,get:[3,4],get_commandline_argu:6,get_config:4,get_num_paramet:4,get_shift:[4,7],git:[1,9],github:[1,4],given:4,global:4,globalaffinelay:4,globalresc:4,h_k:4,half:4,handl:4,handler:4,has:4,hast:4,have:[0,4,9],heatmap:4,height:4,helper:4,henc:4,hep:4,here:4,hidden_shap:4,highli:0,histogram:4,histori:4,hook:4,hopefulli:4,how:4,howev:4,html:[4,9],http:[0,4],ignor:4,imaginari:4,implement:4,includ:[2,4],increas:4,index:[3,4,7,9],indic:4,infer:4,inferr:4,inform:6,init_log:6,init_output:6,initi:4,initialis:[4,6],input:4,input_param:[4,6],insid:4,instal:[2,3,9],instanc:4,instead:4,instruct:1,integ:4,integr:4,intermedi:4,intern:4,interrupt:4,interv:4,introduct:2,invalidcheckpointerror:4,invalidtrainingoutputerror:4,invari:4,invers:4,inverse_pole_mass:4,ising:4,ising_coeffici:4,ising_energi:4,itertool:4,its:0,itself:4,jacobian:4,judg:4,kappa:4,kei:4,keyboard:4,kinet:4,knot:4,known:4,kullbach:4,kwarg:6,label:4,lam:4,lambda:4,land:9,larg:4,lat:4,latent:4,latter:4,lattic:4,lattice_dimens:4,lattice_length:4,lattice_s:4,layer:8,layer_spec:4,learn:4,learnabl:4,leav:4,left:4,leibler:4,length:4,less:4,lexicograph:4,link:3,list:4,load:4,loaded_checkpoint:4,loaded_model:4,loaded_optim:4,loaded_schedul:4,log:[4,6],log_dens:4,logarithm:4,loglevel:6,logrationanerror:4,loop:[4,6],loss:4,loss_sample_interv:4,low:4,low_momentum_correlation_length:4,m_p:4,m_sq:4,mac:4,machineri:4,maco:4,made:4,magnet:4,magnetic_suscept:4,magnetization_autocorr:4,magnetization_integrated_autocorr:4,magnetization_optimal_window:4,magnetization_seri:4,mai:4,main:6,make:[1,9],manag:0,manual_bootstrap_se:4,markov:4,mass:4,match:4,matric:4,matrix:4,maximum:4,mean:[4,7],measur:4,method:[0,4],metropoli:4,metropolis_hast:4,metropolis_test:4,miniconda:0,minimum:4,model:[6,8],model_log_dens:4,model_neg:4,model_to_load:4,modifi:4,modul:[3,8],moment:4,momenta:4,momentum:4,monoton:4,mont:4,more:4,mult:4,multipl:4,multipli:4,multiprocess:4,must:4,n_addit:4,n_affin:4,n_batch:4,n_boot:4,n_config:4,n_sampl:4,n_segment:4,n_spline:4,name:6,navig:[1,3,9],ndarrai:4,nearest:4,necessari:4,need:[4,9],neg:4,negative_mag:4,neighbour:4,network:4,neural:4,nflow:4,nice:4,nnpdf:0,no_final_activ:4,node:4,none:[4,6],nonetyp:4,normal:[4,7],normalis:4,notat:4,note:4,num:4,number:[4,7],numer:4,numpi:4,nvp:4,object:4,observ:8,obtain:4,odd:4,onc:9,one:4,onli:4,oper:4,optim:4,optimal_window:4,optimis:4,optimizer_param:4,option:4,order:[4,9],ordereddict:4,org:4,out:4,outpath:4,output:[4,6],output_dict:4,over:4,overridden:4,packag:[0,1,8],page:[3,9],pair:4,param:4,paramet:4,parameteris:4,parit:4,pars:[4,6],parse_bootstrap_sample_s:4,parse_checkpoint:4,parse_coupl:4,parse_cp_id:4,parse_epoch:4,parse_lattice_dimens:4,parse_lattice_length:4,parse_n_batch:4,parse_optim:4,parse_optimizer_param:4,parse_parameteris:4,parse_sample_interv:4,parse_sample_s:4,parse_save_interv:4,parse_schedul:4,parse_scheduler_param:4,parse_sigma:4,parse_therm:4,parse_training_output:4,parse_window:4,parser:6,part:4,partit:4,pass:4,passiv:4,path:4,pattern:4,pdf:[4,6],per:4,perform:4,period:4,phi:4,phi_:4,phi_four:4,phi_k:4,phi_model:4,phi_tild:4,phia:4,phib:4,phifouract:4,phifourscalar:4,pickl:4,piecewis:4,pip:1,pipelin:6,plateau:4,plot:8,plot_bootstrap_effective_pole_mass:4,plot_bootstrap_ising_energi:4,plot_bootstrap_multiple_numb:4,plot_bootstrap_single_numb:4,plot_bootstrap_suscept:4,plot_bootstrap_two_point:4,plot_bootstrap_zero_momentum_2pf:4,plot_effective_pole_mass:4,plot_example_config:4,plot_field_compon:4,plot_kinetic_eigenvalu:4,plot_magnetization_autocorr:4,plot_magnetization_integrated_autocorr:4,plot_magnetization_seri:4,plot_two_point_correl:4,plot_zero_momentum_correl:4,point:[3,4],pole:4,pool:4,posit:4,prediciton:6,predict:4,probabl:4,problem:6,process:4,produc:4,produce_base_dist:4,produce_bootstrap_se:4,produce_checkpoint:4,produce_geometri:4,produce_lattice_s:4,produce_model_act:4,produce_size_half:4,produce_target_dist:4,produce_training_context:4,produce_training_geometri:4,produce_use_multiprocess:4,product:4,program:4,properti:6,proposal_log_ratio:4,provid:[4,6],python:[0,1,4],pytorch:[0,4],quadrat:4,quadratic_coeffici:4,quantiti:4,quartic_coeffici:4,rand:4,random:4,rate:4,ration:4,rational_quadratic_splin:4,rationalquadraticsplinelay:4,reach:4,real:4,real_nvp:4,recip:4,recommend:0,refer:4,reflect:1,regist:4,relat:4,reli:4,remain:4,remov:6,replac:1,repo:[1,9],report:4,reportengin:[4,6],repres:4,represent:4,reproduc:7,requir:4,respect:4,result:4,revers:4,reverse_kl:4,right:4,roll:4,root:[1,9],row:4,run:[0,1,4,6,7,9],runcard:4,s_k:4,s_x:4,same:[4,9],sampl:[6,7,8],sample_interv:4,sample_s:4,sampleapp:6,save:4,save_checkpoint:4,save_int:4,save_interv:4,scalar:[4,6],scale:4,schedul:4,scheduler_param:4,scheme:4,scienc:0,script:[4,8],search:3,second:4,second_moment_correlation_length:4,section:[2,4],see:4,seed:4,segment:4,separ:0,sequenc:4,sequenti:4,set:[0,4],sever:4,shape:4,shift:4,shiftsmismatcherror:4,should:[1,4,9],shown:4,side:4,sigma:[4,7],signum:4,silent:4,simpl:[6,7],simpli:[0,4],simultan:4,sin:4,sinc:4,singl:4,site:4,size:4,size_half:4,size_in:4,size_out:4,slice:4,slope:4,small:4,someth:4,sort:4,space:4,spatial:4,special:6,specif:4,specifi:4,sphinx:9,spline:4,spline_affin:4,split:4,squar:4,stabl:4,stack:4,stage:4,standard:4,start:[3,4,9],state:4,state_2d:4,stationari:4,step:4,still:4,store:4,str:4,string:4,subclass:[4,6],submodul:8,subpackag:8,success:4,suggest:4,suit:7,sum:4,sum_:4,sum_i:4,sum_x:4,suppli:[0,4],support:[0,4],suscept:4,system:4,tabl:8,table_autocorrel:4,table_correlation_length:4,table_effective_pole_mass:4,table_fit:4,table_kinetic_eigenvalu:4,table_magnet:4,table_real_space_vari:4,table_two_point_correl:4,table_two_point_scalar:4,table_zero_momentum_correl:4,tabul:4,take:4,taken:4,tanh:4,target:4,target_dist:4,target_log_dens:4,tau_chain:4,tensor:4,term:4,termin:4,test:[4,8],test_benchmark:[4,8],test_benchmark_run:7,test_checkerboard:7,test_distribut:[4,8],test_geometri:[4,8],test_index:7,test_normal_distribut:7,test_splitcart:7,test_splitlexi:7,them:4,theoret:[4,6],theori:[4,6],therefor:4,therm:4,thermal:4,thi:[0,2,3,4,9],through:4,tild:4,tildephi:4,time:[4,6],toi:4,top:[3,4],torch:4,total:4,train:[6,8],train_rang:4,trainabl:4,trainapp:6,trainconfig:6,trainenv:6,trainerror:6,training_context:4,training_geometri:4,training_output:4,training_upd:4,trainingoutput:4,trainingruncardnotfound:4,transform:4,translat:4,tupl:4,two:4,two_point_connected_correl:4,two_point_correl:4,two_point_iter:4,type:4,unchang:4,unconstrain:4,unexpect:4,unit:[4,7],unnormalis:4,unus:4,updat:[4,6],use:[0,4],use_multiprocess:4,used:[4,6],user:4,using:[0,2,3,4],util:8,v_in:4,valu:[4,7],variabl:4,varianc:4,vector:4,version:0,via:4,view:4,w_k:4,wai:4,welcom:3,when:4,where:4,whether:4,which:4,whilst:4,whose:7,width:4,wilsonmr:0,window:4,wish:1,within:[4,7],without:4,wolff:4,work:4,written:4,x_base:4,x_k:4,yaml:6,yield:4,you:[0,1,3,9],your:1,z2_equivar:4,z2_equivar_splin:4,zero:4,zero_momentum_2pf:4,zero_momentum_correl:4},titles:["Conda installation","Development install","Getting started","anvil documentation","anvil package","anvil.benchmark_config package","anvil.scripts package","anvil.tests package","anvil","Building the documentation"],titleterms:{anvil:[3,4,5,6,7,8],anvil_benchmark:6,anvil_sampl:6,anvil_train:6,api:4,benchmark:4,benchmark_config:5,build:9,check:4,checkpoint:4,conda:0,config:4,content:[3,4,5,6,7],core:4,develop:1,distribut:4,document:[3,9],free_scalar:4,geometri:4,get:2,indic:3,instal:[0,1],layer:4,model:4,modul:[4,5,6,7],observ:4,packag:[4,5,6,7],plot:4,sampl:4,script:6,start:2,submodul:[4,6,7],subpackag:4,tabl:[3,4],test:7,test_benchmark:7,test_distribut:7,test_geometri:7,train:4,util:4}})