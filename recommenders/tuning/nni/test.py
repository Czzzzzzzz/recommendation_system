import nni

#params = {
#    'features': 512,
#    'lr': 0.001,
#    'momentum': 0,
#}

optimized_params = nni.get_next_parameter()
#params.update(optimized_params)
#print(params)# optimized_params = nni.get_next_parameter()

print(optimized_params)

nni.report_final_result(1)