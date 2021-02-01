#include "/home/kyf/Xilinx/Vitis_HLS/2020.2/include/gmp.h"
#include "weight_and_bias.h"
#include "activation.h"
#include "top.h"
template<typename T>
double tanh_lut(T input){
	Limit_t lower_limit = -3.0;
	Limit_t upper_limit = 3.0;
	RecipStep_t recip_step = 42.5;

	double input_temp = input;
	double output;

	// If we are outside of LUT range
	if (input_temp <= lower_limit)
	{
		output = lut_tanh[0];
	}
	else if (input_temp >= upper_limit)
	{
		output = lut_tanh[NUMBER_OF_LUT_ETRIES_TANH_1-1];
	}
	else
	{
		// Scale from [lower, upper] to [0, N]
		double t = input_temp - lower_limit;
		uint16_t index = t * recip_step;

		output = lut_tanh[index];
	}

	return output;
}
template<typename T>
double sigmoid_lut(T input){
	Limit_t lower_limit = -5.0;
	Limit_t upper_limit = 5.0;
	RecipStep_t recip_step = 25.5;

	double input_temp = input;
	double output;

	// If we are outside of LUT range
	if (input_temp <= lower_limit)
	{
		output = lut_sigmoid[0];
	}
	else if (input_temp >= upper_limit)
	{
		output = lut_sigmoid[NUMBER_OF_LUT_ETRIES_SIGMOID_1-1];
	}
	else
	{
		// Scale from [lower, upper] to [0, N]
		double t = input_temp - lower_limit;
		uint16_t index = t * recip_step;

		output = lut_sigmoid[index];
	}

	return output;
}


template<typename T,int W,int H>
void tanh(T (&a)[W][H]){
	tanh_1:for(int i=0;i<W;i++){
//#pragma HLS UNROLL factor=60
		tanh_2:for(int j=0;j<H;j++) {
//#pragma HLS UNROLL factor=60
			a[i][j]=tanh_lut(a[i][j]);
		}
	}
}

template<typename T,int W,int H>
void sigmoid(T (&a)[W][H]){
#pragma HLS ARRAY_PARTITION complete
	sigmoid_1:for(int i=0;i<W;i++){
#pragma HLS UNROLL factor=40
		sigmoid_2:for(int j=0;j<H;j++){
#pragma HLS UNROLL factor=40
			a[i][j]=sigmoid_lut(a[i][j]);
		}
	}
}

template<typename T,typename VT,int H,int BW>
void martix_multi_vector(T (&res)[H][BW],VT* a,T (&b)[H][BW]){
#pragma HLS ARRAY_PARTITION complete
	martix_multi_vector_1:for(int i=0;i<H;i++){
#pragma HLS UNROLL factor=40
		martix_multi_vector_2:for(int j=0;j<BW;j++){
#pragma HLS UNROLL factor=40
				res[i][j]=T(a[i])*b[i][j];
		}
	}
}
template<typename T,int H,int W>
void martix_multi(T (&res)[H][W],T (&a)[H][W],T (&b)[H][W]){
#pragma HLS ARRAY_PARTITION complete
	martix_multi_1:for(int i=0;i<H;i++){
#pragma HLS UNROLL factor=40
		martix_multi_2:for(int j=0;j<W;j++){
#pragma HLS UNROLL factor=40
				res[i][j]=a[i][j]*b[i][j];
		}
	}
}
template<typename T,int AH,int AW,int BH,int BW>
void martix_dot_multi(T (&res)[AH][BW],T (&a)[AH][AW],T (&b)[BH][BW]){
#pragma HLS ARRAY_PARTITION complete
//#pragma HLS PIPELINE
	martix_dot_multi_1:for(int i=0;i<AH;i++){
//#pragma HLS UNROLL factor=60
		martix_dot_multi_2:for(int j=0;j<BW;j++){
//#pragma HLS UNROLL factor=60
			martix_dot_multi_3:for(int k=0;k<AW;k++){
				res[i][j]=res[i][j]+a[i][k]*b[k][j];
			}
		}
	}
}
template<typename T,int H,int W>
void martix_add(T (&a)[H][W],T (&b)[H][W]){
#pragma HLS ARRAY_PARTITION complete
	martix_add_1:for(int i=0;i<H;i++){
#pragma HLS UNROLL factor=40
		martix_add_2:for(int j=0;j<W;j++){
#pragma HLS UNROLL factor=40
				a[i][j]=a[i][j]+b[i][j];
		}
	}
}
template<typename T,int H,int W>
void martix_cp(T (&a)[H][W],T (&b)[H][W]){
#pragma HLS ARRAY_PARTITION complete
	martix_cp_1:for(int i=0;i<H;i++){
#pragma HLS UNROLL factor=40
		martix_cp_2:for(int j=0;j<W;j++){
#pragma HLS UNROLL factor=40
				a[i][j]=b[i][j];
		}
	}
}
template<typename T,int W>
void vector_cp(T a[W],T b[W]){
#pragma HLS ARRAY_PARTITION complete
	vector_cp_1:for(int i=0;i<W;i++){
#pragma HLS UNROLL factor=40
				a[i]=b[i];
		}
	}

void sigmoid_and_dot_multi(KERNEL_TYPE (&x)[INPUT_FEATURE][UNITS],
		RE_KERNEL_TYPE (&h_tmp)[INPUT_FEATURE][UNITS],
		RE_KERNEL_TYPE (&re_kernel)[UNITS][UNITS]){
	martix_dot_multi<RE_KERNEL_TYPE,INPUT_FEATURE,UNITS,UNITS,UNITS>(h_tmp,h_tmp,re_kernel);
	martix_add<KERNEL_TYPE,INPUT_FEATURE,UNITS>(h_tmp,x);
	sigmoid<KERNEL_TYPE,INPUT_FEATURE,UNITS>(h_tmp);
}
RE_KERNEL_TYPE h_tmp_i[INPUT_FEATURE][UNITS];
RE_KERNEL_TYPE h_tmp_f[INPUT_FEATURE][UNITS];
RE_KERNEL_TYPE h_tmp_c[INPUT_FEATURE][UNITS];
RE_KERNEL_TYPE h_tmp_o[INPUT_FEATURE][UNITS];
RE_KERNEL_TYPE c_tmp[INPUT_FEATURE][UNITS];
KERNEL_TYPE x_i[INPUT_FEATURE][UNITS];
KERNEL_TYPE x_f[INPUT_FEATURE][UNITS];
KERNEL_TYPE x_c[INPUT_FEATURE][UNITS];
KERNEL_TYPE x_o[INPUT_FEATURE][UNITS];
KERNEL_TYPE lstm_output[INPUT_FEATURE][UNITS];

void dataflow_1(double input1[],
		double input2[],
		double input3[],
		double input4[],
		KERNEL_TYPE (&x_i)[INPUT_FEATURE][UNITS],
		KERNEL_TYPE (&x_f)[INPUT_FEATURE][UNITS],
		KERNEL_TYPE (&x_c)[INPUT_FEATURE][UNITS],
		KERNEL_TYPE (&x_o)[INPUT_FEATURE][UNITS],
		KERNEL_TYPE (&m_kernel_i)[INPUT_FEATURE][UNITS],
		KERNEL_TYPE (&m_kernel_f)[INPUT_FEATURE][UNITS],
		KERNEL_TYPE (&m_kernel_c)[INPUT_FEATURE][UNITS],
		KERNEL_TYPE (&m_kernel_o)[INPUT_FEATURE][UNITS]){

#pragma HLS DATAFLOW
	martix_multi_vector<KERNEL_TYPE,double,INPUT_FEATURE,UNITS>(x_i,input1,m_kernel_i);
	martix_multi_vector<KERNEL_TYPE,double,INPUT_FEATURE,UNITS>(x_f,input2,m_kernel_f);
	martix_multi_vector<KERNEL_TYPE,double,INPUT_FEATURE,UNITS>(x_c,input3,m_kernel_c);
	martix_multi_vector<KERNEL_TYPE,double,INPUT_FEATURE,UNITS>(x_o,input4,m_kernel_o);
}
void dataflow_2(KERNEL_TYPE (&x_i)[INPUT_FEATURE][UNITS],
		BIAS_TYPE (&m_bias_i)[INPUT_FEATURE][UNITS],
		KERNEL_TYPE (&x_f)[INPUT_FEATURE][UNITS],
		BIAS_TYPE (&m_bias_f)[INPUT_FEATURE][UNITS],
		KERNEL_TYPE (&x_c)[INPUT_FEATURE][UNITS],
		BIAS_TYPE (&m_bias_c)[INPUT_FEATURE][UNITS],
		KERNEL_TYPE (&x_o)[INPUT_FEATURE][UNITS],
		BIAS_TYPE (&m_bias_o)[INPUT_FEATURE][UNITS]){
#pragma HLS DATAFLOW
	martix_add<KERNEL_TYPE,INPUT_FEATURE,UNITS>(x_i,m_bias_i);
	martix_add<KERNEL_TYPE,INPUT_FEATURE,UNITS>(x_f,m_bias_f);
	martix_add<KERNEL_TYPE,INPUT_FEATURE,UNITS>(x_c,m_bias_c);
	martix_add<KERNEL_TYPE,INPUT_FEATURE,UNITS>(x_o,m_bias_o);
}
void dataflow_3(KERNEL_TYPE (&x_i)[INPUT_FEATURE][UNITS],
		RE_KERNEL_TYPE (&h_tmp_i)[INPUT_FEATURE][UNITS],
		RE_KERNEL_TYPE (&m_re_kernel_i)[UNITS][UNITS],
		KERNEL_TYPE (&x_f)[INPUT_FEATURE][UNITS],
		RE_KERNEL_TYPE (&h_tmp_f)[INPUT_FEATURE][UNITS],
		RE_KERNEL_TYPE (&m_re_kernel_f)[UNITS][UNITS],
		KERNEL_TYPE (&x_o)[INPUT_FEATURE][UNITS],
		RE_KERNEL_TYPE (&h_tmp_o)[INPUT_FEATURE][UNITS],
		RE_KERNEL_TYPE (&m_re_kernel_o)[UNITS][UNITS]){
#pragma HLS DATAFLOW
	sigmoid_and_dot_multi(x_i,h_tmp_i,m_re_kernel_i);
	sigmoid_and_dot_multi(x_f,h_tmp_f,m_re_kernel_f);
	sigmoid_and_dot_multi(x_o,h_tmp_o,m_re_kernel_o);
}
void lstm_cell(double* input,bool isOutput){
/*
	 * x_i = inputs * kernel_i
	 * x_f = inputs * kernel_f
	 * x_c = inputs * kernel_c
	 * x_o = inputs * kernel_o
	 */
	double input_1[INPUT_FEATURE];
#pragma HLS BIND_STORAGE variable=input_1 type=ram_s2p
	double input_2[INPUT_FEATURE];
#pragma HLS BIND_STORAGE variable=input_2 type=ram_s2p
	double input_3[INPUT_FEATURE];
#pragma HLS BIND_STORAGE variable=input_3 type=ram_s2p
	double input_4[INPUT_FEATURE];
#pragma HLS BIND_STORAGE variable=input_4 type=ram_s2p

	vector_cp<double,INPUT_FEATURE>(input_1,input);
	vector_cp<double,INPUT_FEATURE>(input_2,input);
	vector_cp<double,INPUT_FEATURE>(input_3,input);
	vector_cp<double,INPUT_FEATURE>(input_4,input);

	dataflow_1(input_1,input_2,input_3,input_4,
			x_i,x_f,x_c,x_o,
			m_kernel_i,m_kernel_f,m_kernel_c,m_kernel_o);

	/*
	 * x_i += bias_i
	 * x_f += bias_f
	 * x_c += bias_c
	 * x_o += bias_o
	 */
	dataflow_2(x_i,m_bias_i,x_f,m_bias_f,x_c,m_bias_c,x_o,m_bias_o);
//	//i = hard_sigmoid(x_i + np.dot(h_tm_i, recurrent_kernel_i))
//	martix_dot_multi<INPUT_FEATURE,UNITS,UNITS,UNITS>(h_tmp_i,h_tmp_i,m_re_kernel_i);
//	martix_add<INPUT_FEATURE,UNITS>(h_tmp_i,x_i);
//	sigmoid<INPUT_FEATURE,UNITS>(h_tmp_i);
//
//	//f = hard_sigmoid(x_f + np.dot(h_tm_f, recurrent_kernel_f))
//	martix_dot_multi<INPUT_FEATURE,UNITS,UNITS,UNITS>(h_tmp_f,h_tmp_f,m_re_kernel_f);
//	martix_add<INPUT_FEATURE,UNITS>(h_tmp_f,x_f);
//	sigmoid<INPUT_FEATURE,UNITS>(h_tmp_f);
//
//	//o = hard_sigmoid(x_o + np.dot(h_tm_o, recurrent_kernel_o))
////	double temp_o[INPUT_FEATURE][UNITS];
//	martix_dot_multi<INPUT_FEATURE,UNITS,UNITS,UNITS>(h_tmp_o,h_tmp_o,m_re_kernel_o);
//	martix_add<INPUT_FEATURE,UNITS>(h_tmp_o,x_o);
//	sigmoid<INPUT_FEATURE,UNITS>(h_tmp_o);

	dataflow_3(x_i,h_tmp_i,m_re_kernel_i,x_f,h_tmp_f,m_re_kernel_f,x_o,h_tmp_o,m_re_kernel_o);

//	if(isOutput){
//		martix_cp<KERNEL_TYPE,INPUT_FEATURE,UNITS>(lstm_output,h_tmp_o);
//	}
	martix_cp<KERNEL_TYPE,INPUT_FEATURE,UNITS>(lstm_output,h_tmp_o);
	// c = f * c_tm + i * np.tanh(x_c + np.dot(h_tm_c, recurrent_kernel_c))
	martix_dot_multi<KERNEL_TYPE,INPUT_FEATURE,UNITS,UNITS,UNITS>(h_tmp_c,h_tmp_c,m_re_kernel_c);//INPUT_FEATURE X UNITS
	martix_add<KERNEL_TYPE,INPUT_FEATURE,UNITS>(h_tmp_c,x_c);//INPUT_FEATURE X UNITS
	tanh<KERNEL_TYPE,INPUT_FEATURE,UNITS>(h_tmp_c);//INPUT_FEATURE X UNITS
//	double temp_c1[INPUT_FEATURE][UNITS];
	martix_multi<KERNEL_TYPE,INPUT_FEATURE,UNITS>(h_tmp_c,h_tmp_i,h_tmp_c);//INPUT_FEATURE X UNITS
//	double temp_c2[INPUT_FEATURE][UNITS];
	martix_multi<KERNEL_TYPE,INPUT_FEATURE,UNITS>(c_tmp,h_tmp_f,c_tmp);//INPUT_FEATURE X UNITS
	martix_add<KERNEL_TYPE,INPUT_FEATURE,UNITS>(h_tmp_c,c_tmp);//INPUT_FEATURE X UNITS



	//h = o * np.tanh(c)
	tanh<KERNEL_TYPE,INPUT_FEATURE,UNITS>(h_tmp_c);
	martix_multi<KERNEL_TYPE,INPUT_FEATURE,UNITS>(h_tmp_o,h_tmp_o,h_tmp_c);

	martix_cp<RE_KERNEL_TYPE,INPUT_FEATURE,UNITS>(c_tmp,h_tmp_c);
	martix_cp<RE_KERNEL_TYPE,INPUT_FEATURE,UNITS>(h_tmp_i,h_tmp_o);
	martix_cp<RE_KERNEL_TYPE,INPUT_FEATURE,UNITS>(h_tmp_f,h_tmp_o);
//	martix_cp<INPUT_FEATURE,UNITS>(h_tmp_o,h_tmp_o);
	martix_cp<RE_KERNEL_TYPE,INPUT_FEATURE,UNITS>(h_tmp_c,h_tmp_o);

}
template<int INPUT_W,int INPUT_H,int DENSE_UNITS>
void dense(KERNEL_TYPE (&input)[INPUT_W][INPUT_H],
		KERNEL_TYPE (&kernel)[INPUT_H][DENSE_UNITS],
		BIAS_TYPE  (&bias)[INPUT_W][DENSE_UNITS],
		KERNEL_TYPE (&output)[INPUT_W][DENSE_UNITS]){
	martix_dot_multi<KERNEL_TYPE,INPUT_W,INPUT_H,INPUT_H,DENSE_UNITS>(output,input,kernel);
	martix_add<KERNEL_TYPE,INPUT_W,DENSE_UNITS>(output,bias);
	sigmoid<KERNEL_TYPE,INPUT_W,DENSE_UNITS>(output);
}

void single_model(double* input,double* output,int timestep){

lstm_ts:for(int ts=0;ts<INPUT_FEATURE*timestep;ts+=INPUT_FEATURE){
#pragma HLS LOOP_TRIPCOUNT avg=34 max=34 min=34
		lstm_cell(input+ts,false);
	}
//	lstm_cell(input+INPUT_FEATURE*(timestep-1),true);
	KERNEL_TYPE dense1_output[INPUT_FEATURE][DENSE1_UNITS];
	KERNEL_TYPE dense2_output[INPUT_FEATURE][DENSE2_UNITS];
	dense<INPUT_FEATURE,UNITS,DENSE1_UNITS>(lstm_output,dense1_kernel,dense1_bias,dense1_output);
	dense<INPUT_FEATURE,DENSE1_UNITS,DENSE2_UNITS>(dense1_output,dense2_kernel,dense2_bias,dense2_output);
	*output=dense2_output[0][0];
}
void top(double *first_input,double * second_input,double *output,int timestep){
#pragma HLS ARRAY_PARTITION variable=lstm_output complete
#pragma HLS ARRAY_PARTITION variable=c_tmp complete
#pragma HLS ARRAY_PARTITION variable=x_o complete
#pragma HLS ARRAY_PARTITION variable=h_tmp_o complete
#pragma HLS ARRAY_PARTITION variable=x_c complete
#pragma HLS ARRAY_PARTITION variable=h_tmp_i complete
#pragma HLS ARRAY_PARTITION variable=x_f complete
#pragma HLS ARRAY_PARTITION variable=h_tmp_c complete
#pragma HLS ARRAY_PARTITION variable=x_i complete
#pragma HLS ARRAY_PARTITION variable=h_tmp_f complete
#pragma HLS BIND_STORAGE variable=x_o type=ram_t2p
#pragma HLS BIND_STORAGE variable=x_i type=ram_t2p
#pragma HLS BIND_STORAGE variable=x_f type=ram_t2p
#pragma HLS BIND_STORAGE variable=c_tmp type=ram_t2p
#pragma HLS BIND_STORAGE variable=x_c type=ram_t2p
#pragma HLS BIND_STORAGE variable=lstm_output type=ram_t2p
#pragma HLS BIND_STORAGE variable=h_tmp_i type=ram_t2p
#pragma HLS BIND_STORAGE variable=h_tmp_c type=ram_t2p
#pragma HLS BIND_STORAGE variable=h_tmp_o type=ram_t2p
#pragma HLS BIND_STORAGE variable=h_tmp_f type=ram_t2p
	double first_output;
	single_model(first_input,&first_output,timestep);
	*(second_input+timestep+1)=first_output;
	double second_output;
	single_model(second_input,&second_output,timestep+2);
	*output=second_output;
}
