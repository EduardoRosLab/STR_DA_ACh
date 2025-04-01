/***************************************************************************
 *                           AsabukiSynapseWeightChange.cpp                *
 *                           -------------------                           *
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

#include "../../include/learning_rules/AsabukiSynapseWeightChange.h"

#include "../../include/learning_rules/AsabukiSynapseState.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/Utils.h"

#include "../../include/neuron_model/NeuronState.h"

#include <iostream>

AsabukiSynapseWeightChange::AsabukiSynapseWeightChange()
	: WithPostSynaptic(), target_state()
{
	// Set the default values for the learning rule parameters
	this->SetParameters(AsabukiSynapseWeightChange::GetDefaultParameters());
	this->GenerateTableExp();
}

AsabukiSynapseWeightChange::~AsabukiSynapseWeightChange() {}

void AsabukiSynapseWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons)
{
	this->State = (ConnectionState *)new AsabukiSynapseState(
		NumberOfSynapses, 
		this->is_lateral,
		this->tau_p,
		this->tau_d,
		this->Cp,
		this->Cd);
}

void AsabukiSynapseWeightChange::GenerateTableExp()
{
	for (int i=0; i<TableSizeExp; i++) {
	    float x = (i/float(TableSizeExp-1))*(exp_x_max-exp_x_min) + exp_x_min;
	    // i = ((x-exp_x_min)/(exp_x_max-exp_x_min))*(TableSizeExp-1);
		texp[i] = expf(x);
	}
}

float AsabukiSynapseWeightChange::get_exp(float x) {
	int i = ((x-exp_x_min)/(exp_x_max-exp_x_min))*(TableSizeExp-1);
	if (i<0) i = 0;
	if (i>=TableSizeExp) i = TableSizeExp-1;
	return texp[i]; 
}

void AsabukiSynapseWeightChange::ApplyPreSynapticSpike(Interconnection *Connection, double SpikeTime)
{
	if (this->is_lateral==1)
	{
		unsigned int LearningRuleIndex = Connection->GetLearningRuleIndex_withPost();

		// Apply synaptic activity decaying rule
		State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

		// Apply presynaptic spike
		State->ApplyPresynapticSpike(LearningRuleIndex);

		// Apply weight change
		Connection->IncrementWeight(this->eta * State->GetPostsynapticActivity(LearningRuleIndex));
	}
	else 
	{
		int LearningRuleIndex = Connection->GetLearningRuleIndex_withPost();
		double last_spike_time = State->GetLastUpdateTime(LearningRuleIndex);
		State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);
		State->ApplyPresynapticSpike(LearningRuleIndex);

		float n_state = target_state[0];
		float w = Connection->GetWeight();
		//float dif_time = 1e3*(1.0l - get_exp(last_spike_time-SpikeTime));
		//float dif_time = 1e3*(1.0l - expf(last_spike_time-SpikeTime));
		//float dif_w = eta*(n_state*tau - gamma*w*dif_time);

		float dif_w = eta*(n_state*tau - gamma*w);
		Connection->IncrementWeight(dif_w);
	}
}

void AsabukiSynapseWeightChange::ApplyPostSynapticSpike(Neuron * neuron, double SpikeTime){
	if (this->is_lateral==1) {
		unsigned int LearningRuleId = this->GetLearningRuleIndex();
		for (int i = 0; i<neuron->GetInputNumberWithPostSynapticLearning(LearningRuleId); ++i){
			Interconnection * interi = neuron->GetInputConnectionWithPostSynapticLearningAt(LearningRuleId,i);

			unsigned int LearningRuleIndex = neuron->IndexInputLearningConnections[0][LearningRuleId][i];

			// Apply synaptic activity decaying rule
			State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, true);

			// Apply postsynaptic spike
			State->ApplyPostsynapticSpike(LearningRuleIndex);

			// Update synaptic weight
			interi->IncrementWeight(this->eta * State->GetPresynapticActivity(LearningRuleIndex));
		}
	}
}


ModelDescription AsabukiSynapseWeightChange::ParseLearningRule(FILE *fh) 
{
	ModelDescription lrule;

	// float ltauLTP, lMaxChangeLTP, ltauLTD, lMaxChangeLTD;
	// if (fscanf(fh, "%f", &lMaxChangeLTP) != 1 ||
	// 	fscanf(fh, "%f", &ltauLTP) != 1 ||
	// 	fscanf(fh, "%f", &lMaxChangeLTD) != 1 ||
	// 	fscanf(fh, "%f", &ltauLTD) != 1)
	// {
	// 	throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_LEARNING_RULE_LOAD, REPAIR_STDP_WEIGHT_CHANGE_LOAD);
	// }
	// if (ltauLTP <= 0 || ltauLTD <= 0)
	// {
	// 	throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_STDP_WEIGHT_CHANGE_TAUS, REPAIR_LEARNING_RULE_VALUES);
	// }

	// lrule.model_name = AsabukiSynapseWeightChange::GetName();
	// lrule.param_map["max_LTP"] = boost::any(lMaxChangeLTP);
	// lrule.param_map["tau_LTP"] = boost::any(ltauLTP);
	// lrule.param_map["max_LTD"] = boost::any(lMaxChangeLTD);
	// lrule.param_map["tau_LTD"] = boost::any(ltauLTD);

	return lrule;
}

void AsabukiSynapseWeightChange::SetParameters(std::map<std::string, boost::any> param_map) 
{

	// Search for the parameters in the dictionary
	std::map<std::string, boost::any>::iterator it;
	
	it = param_map.find("eta");
	if (it != param_map.end())
	{
		this->eta = boost::any_cast<float>(it->second);
		param_map.erase(it);
	}

	it = param_map.find("tau");
	if (it != param_map.end())
	{
		this->tau = boost::any_cast<float>(it->second);
		param_map.erase(it);
	}

	it = param_map.find("gamma");
	if (it != param_map.end())
	{
		this->gamma = boost::any_cast<float>(it->second);
		param_map.erase(it);
	}

	it = param_map.find("is_lateral");
	if (it != param_map.end())
	{
		this->is_lateral = boost::any_cast<int>(it->second);
		param_map.erase(it);
	}

	it = param_map.find("tau_p");
	if (it != param_map.end())
	{
		this->tau_p = boost::any_cast<float>(it->second);
		param_map.erase(it);
	}

	it = param_map.find("tau_d");
	if (it != param_map.end())
	{
		this->tau_d = boost::any_cast<float>(it->second);
		param_map.erase(it);
	}

	it = param_map.find("Cp");
	if (it != param_map.end())
	{
		this->Cp = boost::any_cast<float>(it->second);
		param_map.erase(it);
	}

	it = param_map.find("Cd");
	if (it != param_map.end())
	{
		this->Cd = boost::any_cast<float>(it->second);
		param_map.erase(it);
	}

	WithPostSynaptic::SetParameters(param_map);
}

ostream &AsabukiSynapseWeightChange::PrintInfo(ostream &out)
{
	out << "- AsabukiSynapse Learning Rule: " << endl;
	out << "\t eta:" << this->eta << endl;
	out << "\t tau:" << this->tau << endl;
	out << "\t gamma:" << this->gamma << endl;

	out << "\t is_lateral:" << this->is_lateral << endl;
	out << "\t tau_p:" << this->tau_p << endl;
	out << "\t tau_d:" << this->tau_d << endl;
	out << "\t Cp:" << this->Cp << endl;
	out << "\t Cd:" << this->Cd << endl;

	return out;
}


void AsabukiSynapseWeightChange::SetTargetState(float *new_target_state)
{
	this->target_state = new_target_state;
}

LearningRule *AsabukiSynapseWeightChange::CreateLearningRule(ModelDescription lrDescription)
{
	AsabukiSynapseWeightChange *lrule = new AsabukiSynapseWeightChange();
	lrule->SetParameters(lrDescription.param_map);
	return lrule;
}

std::map<std::string, boost::any> AsabukiSynapseWeightChange::GetParameters()
{
	std::map<std::string, boost::any> newMap = WithPostSynaptic::GetParameters();
	
	newMap["eta"] = boost::any(this->eta);
	newMap["tau"] = boost::any(this->tau);
	newMap["gamma"] = boost::any(this->gamma);

	newMap["is_lateral"] = boost::any(this->is_lateral);
	newMap["tau_p"] = boost::any(this->tau_p);
	newMap["tau_d"] = boost::any(this->tau_d);
	newMap["Cp"] = boost::any(this->Cp);
	newMap["Cd"] = boost::any(this->Cd);

	return newMap;
}

std::map<std::string, boost::any> AsabukiSynapseWeightChange::GetDefaultParameters()
{
	std::map<std::string, boost::any> newMap = WithPostSynaptic::GetDefaultParameters();
	
	newMap["eta"] = boost::any(5e-6f);
	newMap["tau"] = boost::any(5.0f);
	newMap["gamma"] = boost::any(5.0f);

	newMap["is_lateral"] = boost::any(0);
	newMap["tau_p"] = boost::any(0.020f);
	newMap["tau_d"] = boost::any(0.040f);
	newMap["Cp"] = boost::any(-0.0105f);
	newMap["Cd"] = boost::any(0.00525f);

	return newMap;
}
