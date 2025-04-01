/***************************************************************************
 *                           AsabukiNeuron.h                               *
 *                           ---------------                               *
 * copyright            : (C) 2022 by Álvaro González-Redondo              *
 * email                : alvarogr@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "neuron_model/AsabukiNeuron.h"
#include "neuron_model/VectorNeuronState.h"
#include "neuron_model/CurrentSynapseModel.h"

#include "spike/Neuron.h"
#include "spike/Interconnection.h"

#include "integration_method/IntegrationMethodFactory.h"

#include "../../include/learning_rules/AsabukiSynapseWeightChange.h"


//void update_online_mean(float& mean, float& M2, float new_value, float weight) __attribute__((optimize(0)));

void update_online_mean(float& mean, float& M2, float new_value, float weight)
{
	float delta = new_value - mean;
	mean += delta * weight;

	float delta2 = new_value - mean;
	M2 += delta * delta2;
	M2 -= M2 * weight;

	// M2   = min(1e3f, max(1e-3f, M2));
	// mean = min(1e2f, max(1e-9f, mean));
}

float get_online_var(float M2, float weight) {
	return M2 * weight;
}

void AsabukiNeuron::GenerateTableExp()
{
	for (int i=0; i<TableSizeExp; i++) {
	    float x = (i/float(TableSizeExp-1))*(exp_x_max-exp_x_min) + exp_x_min;
	    // i = ((x-exp_x_min)/(exp_x_max-exp_x_min))*(TableSizeExp-1);
		texp[i] = expf(x);
	}
}

float AsabukiNeuron::get_exp(float x) {
	int i = ((x-exp_x_min)/(exp_x_max-exp_x_min))*(TableSizeExp-1);
	if (i<0) i = 0;
	if (i>=TableSizeExp) i = TableSizeExp-1;
	return texp[i]; 
}

void AsabukiNeuron::Generate_g_nmda_inf_values()
{
	auxNMDA = (TableSizeNMDA - 1);
	for (int i = 0; i < TableSizeNMDA; i++)
	{
		float V = -100.0 + ((100.0 - 100.0) * i) / (TableSizeNMDA - 1);

		// g_nmda_inf
		g_nmda_inf_values[i] = 1.0f / (1.0f + exp(-0.062f * V) * (1.2f / 3.57f));
	}
}

float AsabukiNeuron::Get_g_nmda_inf(float V_m)
{
	int position = int((V_m - 100.0) * auxNMDA);
	if (position < 0)
	{
		position = 0;
	}
	else if (position > (TableSizeNMDA - 1))
	{
		position = TableSizeNMDA - 1;
	}
	return g_nmda_inf_values[position];
}

void AsabukiNeuron::InitializeCurrentSynapsis(int N_neurons)
{
	this->CurrentSynapsis = new CurrentSynapseModel(N_neurons);
}

// this neuron model is implemented in a second scale.
AsabukiNeuron::AsabukiNeuron()
	: TimeDrivenNeuronModel(MilisecondScale), EXC(false), INH(false), NMDA(false), EXT_I(false), 
	tau_exc(1), inv_tau_exc(1), tau_inh(1), inv_tau_inh(1), tau_nmda(1), inv_tau_nmda(1), tau(1), g_l(1), tau_ref(1), g_d(1), alpha(1), t_0(1), inv_t_0(1), beta_0(1), theta_0(1), exp_bt(1), phi_0(1)
{	

	std::map<std::string, boost::any> param_map = AsabukiNeuron::GetDefaultParameters();
	param_map["name"] = AsabukiNeuron::GetName();
	this->SetParameters(param_map);

	this->State = (VectorNeuronState *)new VectorNeuronState(N_NeuronStateVariables, true);

	this->GenerateTableExp();
}

AsabukiNeuron::~AsabukiNeuron(void) { }

VectorNeuronState *AsabukiNeuron::InitializeState()
{
	return this->GetVectorNeuronState();
}

InternalSpike *AsabukiNeuron::ProcessInputSpike(Interconnection *inter, double time)
{
	// Add the effect of the input spike
	uint target_neuron = inter->GetTargetNeuronModelIndex();
	int position = N_DifferentialNeuronState + inter->GetType();
	float increment = inter->GetWeight();
	this->GetVectorNeuronState()->IncrementStateVariableAtCPU(target_neuron, position, increment);

	return 0;
}

void AsabukiNeuron::ProcessInputCurrent(Interconnection *inter, Neuron *target, float current)
{
	// Update the external current in the corresponding input synapse of type EXT_I (defined in pA).
	this->CurrentSynapsis->SetInputCurrent(target->GetIndex_VectorNeuronState(), inter->GetSubindexType(), current);

	// Update the total external current that receive the neuron coming from all its EXT_I synapsis (defined in pA).
	float total_ext_I = this->CurrentSynapsis->GetTotalCurrent(target->GetIndex_VectorNeuronState());
	State->SetStateVariableAt(target->GetIndex_VectorNeuronState(), EXT_I_index, total_ext_I);
}

bool AsabukiNeuron::UpdateState(int index, double CurrentTime)
{
	// Reset the number of internal spikes in this update period
	this->State->NInternalSpikeIndexs = 0;

	this->integration_method->NextDifferentialEquationValues();

	this->CheckValidIntegration(CurrentTime, this->integration_method->GetValidIntegrationVariable());

	return false;
}

enum NeuronModelOutputActivityType AsabukiNeuron::GetModelOutputActivityType()
{
	return OUTPUT_SPIKE;
}

enum NeuronModelInputActivityType AsabukiNeuron::GetModelInputActivityType()
{
	return INPUT_SPIKE_AND_CURRENT;
}

ostream &AsabukiNeuron::PrintInfo(ostream &out)
{
	out << "- Asabuki et al. (2020) neuron: " << AsabukiNeuron::GetName() << endl;
	out << "\tExcitatory receptor time constant (tau_exc): " << this->tau_exc << "ms" << endl;
	out << "\tInhibitory receptor time constant (tau_inh): " << this->tau_inh << "ms" << endl;
	out << "\tSoma time constant (tau): " << this->tau << "ms" << endl;
	out << "\tConductance between soma and dendrite compartiments (g_d): " << this->g_d << "nS" << endl;
	out << "\tDuration of online mean and variance (t_0): " << this->t_0 << "ms" << endl;
	out << "\t??? (beta_0): " << this->beta_0 << "???" << endl;
	out << "\t??? (theta_0): " << this->theta_0 << "???" << endl;
	out << "\tMaximum firing rate (phi_0): " << this->phi_0 << "Hz" << endl;

	this->integration_method->PrintInfo(out);
	return out;
}

void AsabukiNeuron::InitializeStates(int N_neurons, int OpenMPQueueIndex)
{
	// Initialize neural state variables.
	float initialization[N_NeuronStateVariables] = {0.0f};
	initialization[M2_index] = 1.0f;
	State->InitializeStates(N_neurons, initialization);

	// Initialize integration method state variables.
	this->integration_method->SetBifixedStepParameters(1.0f / 2.0f, 1.0f / 2.0f, 0);
	this->integration_method->Calculate_conductance_exp_values();
	this->integration_method->InitializeStates(N_neurons, initialization);

	// Initialize the array that stores the number of input current synapses for each neuron in the model
	InitializeCurrentSynapsis(N_neurons);
}

void AsabukiNeuron::GetBifixedStepParameters(float &startVoltageThreshold, float &endVoltageThreshold, float &timeAfterEndVoltageThreshold)
{
	startVoltageThreshold = 4 / 5;
	endVoltageThreshold = 4 / 5;
	timeAfterEndVoltageThreshold = 0.0f;
	return;
}

void AsabukiNeuron::EvaluateSpikeCondition(float previous_V, float *NeuronState, int index, float elapsedTimeInNeuronModelScale)
{
	if (NeuronState[V_m_index] > 1.0f)
	{
		NeuronState[V_m_index] = 0.0f;
		State->NewFiredSpike(index);
		this->integration_method->resetState(index);
		this->State->InternalSpikeIndexs[this->State->NInternalSpikeIndexs] = index;
		this->State->NInternalSpikeIndexs++;
	}
}

void AsabukiNeuron::EvaluateDifferentialEquation(float *NeuronState, float *AuxNeuronState, int index, float elapsed_time)
{	
	for (int i=0;i<N_NeuronStateVariables;i++)
		AuxNeuronState[i] = 0.0f;

	// Dendritic potential update
	
	NeuronState[v_index] = 0.0;
	NeuronState[v_index] += NeuronState[EXC_index];
	NeuronState[v_index] += NeuronState[NMDA_index];
	NeuronState[v_index] += NeuronState[EXT_I_index]; // (defined in pA).

	// Soma potential update

	AuxNeuronState[u_index] += g_l*(-NeuronState[u_index]);
	AuxNeuronState[u_index] += g_d*(NeuronState[v_index] - NeuronState[u_index]);
	AuxNeuronState[u_index] -= NeuronState[INH_index];

	// Instantaneous soma and dendritic firing rates 

	float mean, M2, new_value;
	float mu, sigma;
	mean = NeuronState[mean_index];
	M2 = NeuronState[M2_index];
	new_value = NeuronState[u_index];
	update_online_mean(mean, M2, new_value, inv_t_0);

	mu = mean;
	sigma = get_online_var(M2, inv_t_0);
	NeuronState[mean_index] = mean;
	NeuronState[M2_index] = M2;

	float beta_i = beta_0 / sigma;
	float theta_i = mu + sigma*theta_0;

	float phi_som = phi_0 / (1.0f + get_exp(beta_i * (theta_i - NeuronState[u_index])));	// float phi_som = phi_0 / (1.0f + expf(beta_i * (theta_i - NeuronState[u_index])));
	AuxNeuronState[V_m_index] = phi_som * 1e-3;

	float v_att = alpha * NeuronState[v_index];
	float phi_den = phi_0 / (1.0f + get_exp(beta_0 * (theta_0 - v_att)));	// float phi_den = phi_0 / (1.0f + expf(beta_0 * (theta_0 - v_att)));

	float psi = (beta_0 * exp_bt) / (exp_bt + get_exp(beta_0*v_att));	// float psi = (beta_0 * exp_bt) / (exp_bt + expf(beta_0*v_att));
	NeuronState[n_index] = psi * ((phi_som-phi_den) / phi_0);
}

void AsabukiNeuron::EvaluateTimeDependentEquation(float *NeuronState, int index, int elapsed_time_index)
{
	float limit = 1e-9;
	float *Conductance_values = this->Get_conductance_exponential_values(elapsed_time_index);

	if (EXC)
	{
		if (NeuronState[EXC_index] < limit)
		{
			NeuronState[EXC_index] = 0.0f;
		}
		else
		{
			NeuronState[EXC_index] *= Conductance_values[0];
		}
	}
	if (INH)
	{
		if (NeuronState[INH_index] < limit)
		{
			NeuronState[INH_index] = 0.0f;
		}
		else
		{
			NeuronState[INH_index] *= Conductance_values[1];
		}
	}
	if (NMDA)
	{
		if (NeuronState[NMDA_index] < limit)
		{
			NeuronState[NMDA_index] = 0.0f;
		}
		else
		{
			NeuronState[NMDA_index] *= Conductance_values[2];
		}
	}

	// cout.precision(5);
	// for (int i=0; i<N_NeuronStateVariables; i++)
	// 	cout << setprecision(5) << setw(8) << NeuronState[i] << "\t";
	// cout << endl;
}

void AsabukiNeuron::Calculate_conductance_exp_values(int index, float elapsed_time)
{
	// excitatory synapse.
	Set_conductance_exp_values(index, 0, expf(-elapsed_time * this->inv_tau_exc));
	// inhibitory synapse.
	Set_conductance_exp_values(index, 1, expf(-elapsed_time * this->inv_tau_inh));
	// nmda synapse.
	Set_conductance_exp_values(index, 2, expf(-elapsed_time * this->inv_tau_nmda));
	// Firing rate update
	Set_conductance_exp_values(index, 3, elapsed_time * 1e-3);
}

bool AsabukiNeuron::CheckSynapseType(Interconnection *connection)
{
	int Type = connection->GetType();
	if (Type < N_TimeDependentNeuronState && Type >= 0)
	{
		// activaty synapse type
		if (Type == 0)
		{
			EXC = true;
		}
		if (Type == 1)
		{
			INH = true;
		}
		if (Type == 2)
		{
			NMDA = true;
		}
		if (Type == 3)
		{
			EXT_I = true;
		}

		NeuronModel *model = connection->GetSource()->GetNeuronModel();
		// Synapse types that process input spikes
		if (Type < N_TimeDependentNeuronState - 1)
		{
			if (model->GetModelOutputActivityType() == OUTPUT_SPIKE)
			{
				return true;
			}
			else
			{
				cout << "Synapses type " << Type << " of neuron model " << AsabukiNeuron::GetName() << " must receive spikes. The source model generates currents." << endl;
				return false;
			}
		}
		// Synapse types that process input current
		if (Type == N_TimeDependentNeuronState - 1)
		{
			if (model->GetModelOutputActivityType() == OUTPUT_CURRENT)
			{
				connection->SetSubindexType(this->CurrentSynapsis->GetNInputCurrentSynapsesPerNeuron(connection->GetTarget()->GetIndex_VectorNeuronState()));
				this->CurrentSynapsis->IncrementNInputCurrentSynapsesPerNeuron(connection->GetTarget()->GetIndex_VectorNeuronState());
				return true;
			}
			else
			{
				cout << "Synapses type " << Type << " of neuron model " << AsabukiNeuron::GetName() << " must receive current. The source model generates spikes." << endl;
				return false;
			}
		}
	}
	else
	{
		cout << "Neuron model " << AsabukiNeuron::GetName() << " does not support input synapses of type " << Type << ". Just defined " << N_TimeDependentNeuronState << " synapses types." << endl;
		return false;
	}
}

std::map<std::string, boost::any> AsabukiNeuron::GetParameters() const
{
	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = TimeDrivenNeuronModel::GetParameters();

	newMap["tau_exc"] = boost::any(this->tau_exc);
	newMap["tau_inh"] = boost::any(this->tau_inh);
	newMap["tau"] = boost::any(this->tau);
	newMap["g_d"] = boost::any(float(this->g_d));
	
	newMap["tau_nmda"] = boost::any(this->tau_nmda);
	newMap["tau_ref"] = boost::any(this->tau_ref);

	newMap["t_0"] = boost::any(float(this->t_0));
	newMap["beta_0"] = boost::any(float(this->beta_0));
	newMap["theta_0"] = boost::any(float(this->theta_0));
	newMap["phi_0"] = boost::any(float(this->phi_0));

	return newMap;
}

std::map<std::string, boost::any> AsabukiNeuron::GetSpecificNeuronParameters(int index) const 
{
	return GetParameters();
}

void AsabukiNeuron::SetParameters(std::map<std::string, boost::any> param_map) 
{

	// Search for the parameters in the dictionary
	std::map<std::string, boost::any>::iterator it;

	it = param_map.find("tau_exc");
	if (it != param_map.end())
	{
		float new_param = boost::any_cast<float>(it->second);
		this->tau_exc = new_param;
		this->inv_tau_exc = 1.0f / new_param;
		param_map.erase(it);
	}

	it = param_map.find("tau_inh");
	if (it != param_map.end())
	{
		float new_param = boost::any_cast<float>(it->second);
		this->tau_inh = new_param;
		this->inv_tau_inh = 1.0f / new_param;
		param_map.erase(it);
	}

	it = param_map.find("tau");
	if (it != param_map.end())
	{
		float new_param = boost::any_cast<float>(it->second);
		this->tau = new_param;
		this->g_l = 1.0f / new_param;
		this->alpha = g_d / (g_d + g_l);
		param_map.erase(it);
	}

	it = param_map.find("g_d");
	if (it != param_map.end())
	{
		float new_param = boost::any_cast<float>(it->second);
		this->g_d = new_param;
		this->alpha = g_d / (g_d + g_l);
		param_map.erase(it);
	}

	it = param_map.find("tau_nmda");
	if (it != param_map.end())
	{
		float new_param = boost::any_cast<float>(it->second);
		this->tau_nmda = new_param;
		this->inv_tau_nmda = 1.0 / this->tau_nmda;
		param_map.erase(it);
	}

	it = param_map.find("tau_ref");
	if (it != param_map.end())
	{
		float new_param = boost::any_cast<float>(it->second);
		this->tau_ref = new_param;
		param_map.erase(it);
	}

	it = param_map.find("t_0");
	if (it != param_map.end())
	{
		float new_param = boost::any_cast<float>(it->second);
		this->t_0 = new_param;
		this->inv_t_0 = 1.0 / this->t_0;
		param_map.erase(it);
	}

	it = param_map.find("beta_0");
	if (it != param_map.end())
	{
		float new_param = boost::any_cast<float>(it->second);
		this->beta_0 = new_param;
		this->exp_bt = exp(this->beta_0*this->theta_0);
		param_map.erase(it);
	}

	it = param_map.find("theta_0");
	if (it != param_map.end())
	{
		float new_param = boost::any_cast<float>(it->second);
		this->theta_0 = new_param;
		this->exp_bt = exp(this->beta_0*this->theta_0);
		param_map.erase(it);
	}

	it = param_map.find("phi_0");
	if (it != param_map.end())
	{
		float new_param = boost::any_cast<float>(it->second);
		this->phi_0 = new_param;
		param_map.erase(it);
	}

	// Search for the parameters in the dictionary
	TimeDrivenNeuronModel::SetParameters(param_map);

	// Set the new g_nmda_inf values based on the e_exc and e_inh parameters
	Generate_g_nmda_inf_values();

	return;
}

IntegrationMethod *AsabukiNeuron::CreateIntegrationMethod(ModelDescription imethodDescription) 
{
	return IntegrationMethodFactory<AsabukiNeuron>::CreateIntegrationMethod(imethodDescription, (AsabukiNeuron *)this);
}

std::map<std::string, boost::any> AsabukiNeuron::GetDefaultParameters()
{
	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = TimeDrivenNeuronModel::GetDefaultParameters<AsabukiNeuron>();
	newMap["tau_exc"] = boost::any(5.0f); // AMPA (excitatory) receptor time constant (ms)
	newMap["tau_inh"] = boost::any(5.0f); // GABA (inhibitory) receptor time constant (ms)
	newMap["tau"] = boost::any(15.0f);	  // Refractory period (ms)
	newMap["tau_nmda"] = boost::any(50.0f);
	newMap["tau_ref"] = boost::any(2.0f);
	newMap["g_d"] = boost::any(0.7f);	  // Leak conductance (nS)
	newMap["t_0"] = boost::any(3000.0f);
	newMap["beta_0"] = boost::any(5.0f);
	newMap["theta_0"] = boost::any(1.7f);
	newMap["phi_0"] = boost::any(10.0f);

	return newMap;
}

NeuronModel *AsabukiNeuron::CreateNeuronModel(ModelDescription nmDescription)
{
	AsabukiNeuron *nmodel = new AsabukiNeuron();
	nmodel->SetParameters(nmDescription.param_map);
	return nmodel;
}

ModelDescription AsabukiNeuron::ParseNeuronModel(std::string FileName) 
{
	FILE *fh;
	ModelDescription nmodel;
	nmodel.model_name = AsabukiNeuron::GetName();
	long Currentline = 0L;
	fh = fopen(FileName.c_str(), "rt");
	if (!fh)
	{
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, Currentline, FileName.c_str());
	}

	Currentline = 1L;

	skip_comments(fh, Currentline);
	// if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f)
	// {
	// 	throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_C_M, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	// }
	// nmodel.param_map["c_m"] = boost::any(param);

	skip_comments(fh, Currentline);
	// if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f)
	// {
		// throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_TAU_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	// }
	// nmodel.param_map["tau_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	// if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f)
	// {
	// 	throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_TAU_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	// }
	// nmodel.param_map["tau_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	// if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f)
	// {
	// 	throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_TAU_REF, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	// }
	// nmodel.param_map["tau_ref"] = boost::any(param);

	skip_comments(fh, Currentline);
	// if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f)
	// {
	// 	throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_G_LEAK, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	// }
	// nmodel.param_map["g_leak"] = boost::any(param);

	skip_comments(fh, Currentline);
	// if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f)
	// {
	// 	throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_TAU_NMDA, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	// }
	// nmodel.param_map["tau_nmda"] = boost::any(param);

	skip_comments(fh, Currentline);
	// try
	// {
	// 	ModelDescription intMethodDescription = TimeDrivenNeuronModel::ParseIntegrationMethod<LIFTimeDrivenModel>(fh, Currentline);
	// 	nmodel.param_map["int_meth"] = boost::any(intMethodDescription);
	// }
	// catch (EDLUTException exc)
	// {
	// 	throw EDLUTFileException(exc, Currentline, FileName.c_str());
	// }

	nmodel.param_map["name"] = boost::any(AsabukiNeuron::GetName());

	fclose(fh);

	return nmodel;
}

std::string AsabukiNeuron::GetName()
{
	return "AsabukiNeuron";
}

std::map<std::string, std::string> AsabukiNeuron::GetNeuronModelInfo()
{
	// Return a dictionary with the parameters
	std::map<std::string, std::string> newMap;
	newMap["info"] = std::string("Asabuki et al. (2020) neuron model.");
	newMap["tau_exc"] = std::string("Excitatory receptor time constant (ms)");
	newMap["tau_inh"] = std::string("Inhibitory receptor time constant (ms)");
	newMap["tau"] = std::string("Soma time constant (ms)");
	newMap["g_d"] = std::string("Conductance between soma and dendrite compartiments (nS)");

	newMap["tau_nmda"] = std::string("NMDA (excitatory) receptor time constant (ms)");
	newMap["tau_ref"] = std::string("Refractory period (ms)");

	newMap["int_meth"] = std::string("Integraton method dictionary (from the list of available integration methods)");


	return newMap;
}

void AsabukiNeuron::CalculateElectricalCouplingSynapseNumber(Interconnection *inter)
{
	// Learning rule parameters
	LearningRule *lr = inter->GetWeightChange_withPost();
	AsabukiSynapseWeightChange *elr = dynamic_cast<AsabukiSynapseWeightChange *>(lr);

	// Check if this connection is of type eprop
	if (!elr)
		return;

	// Neuron state target index
	int TargetIndex = inter->GetTarget()->GetIndex_VectorNeuronState();

	// Neuron state
	float *NeuronState = this->GetVectorNeuronState()->GetStateVariableAt(TargetIndex);
	// Example: float V = NeuronState[V_m_index] - source_V;

	// Saving the neuron state in the learning rule
	elr->SetTargetState(NeuronState);
}

void AsabukiNeuron::InitializeElectricalCouplingSynapseDependencies()
{
}

void AsabukiNeuron::CalculateElectricalCouplingSynapseDependencies(Interconnection *inter)
{
}
