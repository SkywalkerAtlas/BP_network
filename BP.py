import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.ticker import MultipleLocator

def initialize_random_array(size_row,size_col,num_of_input):
	'return an array with random numbers uniformly distributed inside a small range(over -2.4/total_number_of_inputs to 2.4/total_number_of_inputs)'
	array = np.random.rand(size_row,size_col)
	array *= 2
	array -= 1
	array *= 2.4
	array /= num_of_input
	return array
	
def cal_output(input_array,input_weight_array,threshold_array):
	def sigmoid(x):
		try:
			return 1.0/(1.0+math.exp(-x))
		except:
			if(x>0):
				return 1
			else:
				return 0
	result_output_array = np.zeros(threshold_array.shape,dtype=np.float)
	for i in range(threshold_array.size):
		x = (input_array * input_weight_array[:,i]).sum() - threshold_array[0,i]
		result_output_array[0,i] = sigmoid(x)
	return result_output_array

def cal_o_error_gradient(error_array,actual_output_array):
	return actual_output_array*(1-actual_output_array)*error_array

def cal_d_error_gradient(actual_hidden_output_array,output_layer_error_gradient,output_weight_array):
	hidden_layer_error_gradient = np.zeros(actual_hidden_output_array.shape)
	for index in range(hidden_layer_error_gradient.shape[1]):
		hidden_layer_error_gradient[0,index] = actual_hidden_output_array[0,index]*(1-actual_hidden_output_array[0,index])*((output_weight_array[index]*output_layer_error_gradient).sum())
	return hidden_layer_error_gradient

def training(inputs, input_weight_array, output_weight_array, hidden_layer_threshold_array, output_layer_threshold_array, desire_output_array,learning_speed,mode = 'ab',momentum_term_ = 0.95):
	if('b' in mode):
		#print('b')
		momentum_term = momentum_term_
	else:
		#print('no b')
		momentum_term = 0
	error_array = np.ones(desire_output_array.shape)
	squared_errors = (error_array * error_array).sum()
	output_weight_correction = np.zeros(output_weight_array.shape)
	input_weight_correction = np.zeros(input_weight_array.shape)
	hidden_layer_threshold_weight_correction = np.zeros(hidden_layer_threshold_array.shape)
	output_layer_threshold_weight_correction = np.zeros(output_layer_threshold_array.shape)
	learning_rate_log = []
	squared_errors_log = [1]
	times_count = 1
	while(squared_errors>=0.001 and times_count<20000):            ########################   <-----------------error limits is here
		squared_errors = 0
		learning_rate_log.append(learning_speed)
		for ite in range(inputs.shape[0]):
			actual_hidden_output_array = cal_output(inputs[ite],input_weight_array,hidden_layer_threshold_array)
			actual_output_array = cal_output(actual_hidden_output_array,output_weight_array,output_layer_threshold_array)
			error_array = desire_output_array[ite] - actual_output_array
			squared_errors += (error_array * error_array).sum()
			output_layer_error_gradient = cal_o_error_gradient(error_array,actual_output_array)
			for index in range(actual_hidden_output_array.size):
				output_weight_correction[index,:] =momentum_term*output_weight_correction[index,:] + learning_speed*actual_hidden_output_array[0,index]*output_layer_error_gradient
			output_layer_threshold_weight_correction =momentum_term*output_layer_threshold_weight_correction + learning_speed*output_layer_error_gradient*(-1)
			hidden_layer_error_gradient = cal_d_error_gradient(actual_hidden_output_array,output_layer_error_gradient,output_weight_array)
			for index in range(inputs[ite].size):
				input_weight_correction[index,:] =momentum_term*input_weight_correction[index,:] + learning_speed*inputs[ite][index]*hidden_layer_error_gradient
			hidden_layer_threshold_weight_correction =momentum_term*hidden_layer_threshold_weight_correction + learning_speed*(-1)*hidden_layer_error_gradient
			input_weight_array += input_weight_correction
			output_weight_array += output_weight_correction
			hidden_layer_threshold_array += hidden_layer_threshold_weight_correction
			output_layer_threshold_array += output_layer_threshold_weight_correction
		
		if('a' in mode):
			if(squared_errors > (squared_errors_log[-1])*1.04):
				learning_speed *= 0.7
			else:
				learning_speed *= 1.05
		squared_errors_log.append(squared_errors)
		times_count += 1
	return squared_errors_log,learning_rate_log
		

def main():
	x = np.linspace(1,np.pi/2,200)
	y = 2*np.sin(x)
	y -= 0.7   #y = 2*sin(x)-0.7
	
	num_of_hidden_layer = 2
	num_of_input = 1
	num_of_ouput = 1
	learning_speed = 0.1
	
	x_x = np.random.uniform(1,np.pi/2,size = 100)
	x_x = x_x.reshape(-1,1)
	y_y = 2*np.sin(x_x)
	y_y -= 0.7+0.8
	input_weight_array = initialize_random_array(num_of_input,num_of_hidden_layer,num_of_input)
	output_weight_array = initialize_random_array(num_of_hidden_layer,num_of_ouput,num_of_input)
	hidden_layer_threshold_array = initialize_random_array(1,num_of_hidden_layer,num_of_input)
	output_layer_threshold_array = initialize_random_array(1,num_of_ouput,num_of_input)
	
	
	
	avg_diff_momen_epoch = [0.0 for i in range(25)]
	diff_momen = [0.0 for i in range(25)]
	momentum = 0.5
	for index in range(25):
		diff_momen[index] = momentum
		for times in range(100):
			log,l = training(x_x,initialize_random_array(num_of_input,num_of_hidden_layer,num_of_input),initialize_random_array(num_of_hidden_layer,num_of_ouput,num_of_input),initialize_random_array(1,num_of_hidden_layer,num_of_input),initialize_random_array(1,num_of_ouput,num_of_input),y_y,learning_speed,momentum_term_ = momentum)
			avg_diff_momen_epoch[index] = avg_diff_momen_epoch[index]+len(log)
		avg_diff_momen_epoch[index] = avg_diff_momen_epoch[index]/100
		momentum = 0.02 + momentum
		print(diff_momen[index],avg_diff_momen_epoch[index])
		
	fig,ax = plt.subplots()
	ax.set_title('averge epoch on different momentum term')
	ax.set_xlabel('momentum term')
	ax.set_ylabel('Epoch')
	ax.plot(diff_momen,avg_diff_momen_epoch)
	#log_t_w_d ,l_r_l_w_d= training(x_x, input_weight_array.copy(), output_weight_array.copy(), hidden_layer_threshold_array.copy(), output_layer_threshold_array.copy(), y_y,learning_speed,'b')
	#print(len(log_t_w_d))
	
	#log_t ,l_r_l_t= training(x_x, input_weight_array.copy(), output_weight_array.copy(), hidden_layer_threshold_array.copy(), output_layer_threshold_array.copy(), y_y,learning_speed,'a')
	#print(len(log_t))
	
	#log,learning_rate_log= training(x_x,input_weight_array,output_weight_array,hidden_layer_threshold_array,output_layer_threshold_array,y_y,learning_speed)
	#log_array = np.array(log[1:])
	#print(log_array,log_array.size)
	
	#t_inputs = np.random.uniform(1,np.pi/2,size = 20).reshape(-1,1)
	#t_outputs = np.zeros(20).reshape(-1,1)
	#for i in range(t_inputs.size):
	#	t_input = t_inputs[i]
	#	t_outputs[i] = cal_output(cal_output(t_input,input_weight_array,hidden_layer_threshold_array),output_weight_array,output_layer_threshold_array) + 0.8
	
	#fig,axs = plt.subplots(2,2)
	#ax1 = axs[0,0]
	#ax1 = plt.subplot(221)
	#ax1.plot(x,y,color='red',linewidth=2,label='$2sin(x)-0.7$')
	#ax1.plot(t_inputs,t_outputs,'o',color='b',label='test_set')
	#ax1.set_title('Function Image')
	#ax1.set_xlabel('x')
	#ax1.set_ylabel('$f(x)$')
	#ax1.legend(loc = 'upper left')
	
	#ax2 = axs[0,1]
	#ax2 = plt.subplot(222)
	#ax2.set_title('Learning rate for solving $2sin(x)-0.7$')
	#ax2.set_xlabel('Epoch')
	#ax2.set_ylabel('Learning rate')
	#ax2.plot(range(len(l_r_l_w_d)),l_r_l_w_d,'g',label='with momentum',linewidth=2)
	#ax2.plot(range(len(l_r_l_t)),l_r_l_t,'c',label='with adaptive learning rate',linewidth=2)
	#ax2.plot(range(len(learning_rate_log)),learning_rate_log,'k',label='with momentum&adaptive learning rate',linewidth=2)
	#ax2.legend()
	
	#ax3 = axs[1,1]
	#ax3 = plt.subplot(212)
	#ax3.plot(np.arange(log_array.size),log_array,'k',label='with momentum&adaptive learning rate',linewidth=2)
	#ax3.plot(range(len(log_t_w_d)),log_t_w_d,'g',label='with momentum',linewidth=2)
	#ax3.plot(range(len(log_t)),log_t,'c',label='with adaptive learning rate',linewidth=2)
	#ax3.plot(np.linspace(0,8654,log_array.size),log_array,'g',label='with momentum',linewidth=2)
	#ax3.set_xlabel('Epoch')
	#ax3.set_ylabel('Sum-squared error')
	#ax3.set_title('Learning curve for solving $2sin(x)-0.7$')
	#ax3.set_ylim(0.0001,100)
	#ax3.set_yscale('log')
	#ax3.legend()
	plt.show()

if __name__ == '__main__':
	main()
