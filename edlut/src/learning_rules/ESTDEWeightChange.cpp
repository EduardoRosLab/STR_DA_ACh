/***************************************************************************
 *                           ESTDEWeightChange.cpp                          *
 *                           -------------------                           *
 * copyright            : (C) 2022 by Álvaro González-Redondo                        *
 * email                : alvarogr@ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/learning_rules/ESTDEWeightChange.h"

#include "../../include/learning_rules/ESTDEState.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/Utils.h"

#include "../../include/neuron_model/NeuronState.h"


ESTDEWeightChange::ESTDEWeightChange():WithPostSynaptic(){
	// Set the default values for the learning rule parameters
	this->SetParameters(ESTDEWeightChange::GetDefaultParameters());
}

ESTDEWeightChange::~ESTDEWeightChange(){

}


void ESTDEWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons){
	this->State=(ConnectionState *) new ESTDEState(
		NumberOfSynapses
		, this->tau_pre
		, this->tau_pos
		, this->tau_eli
		, this->tau_ach);
}


void ESTDEWeightChange::ApplyPreSynapticSpike(Interconnection * Connection, double SpikeTime){

	// Not a trigger spike, use normal learning
	if (!Connection->GetTriggerConnection()) {
		unsigned int LearningRuleIndex = Connection->GetLearningRuleIndex_withPost();
		// Get current neuromodulator values
		float ach = State->GetStateVariableAt(LearningRuleIndex, ESTDEState::ACH_INDEX);
		// float ach = 1.0;
		ach = (ach>1.0) ? 1.0 : ach;
		// ach = 1.0 - ach;

		// Apply synaptic activity decaying rule
		State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);
		// Apply presynaptic spike
		State->ApplyPresynapticSpike(LearningRuleIndex);
		// Apply weight change
		// float wdif = ach * this->pospre_dif * State->GetPostsynapticActivity(LearningRuleIndex);
		float wdif = this->pospre_dif * State->GetPostsynapticActivity(LearningRuleIndex) + this->pre_dif;
		Connection->IncrementWeight(wdif);
	}

	// Trigger spike, use special learning rules
	else
	{
		// Check which type of trigger is it
		int con_type = Connection->GetType();

		// Get the pre-pos learning kernel and DA/ACh factors
		float prepos_dif = this->prepos_tri_dif[con_type];
		float pospre_dif = this->pospre_tri_dif[con_type];
		float ach_dif = this->ach_tri_dif[con_type];
		// float ach_dif = 0.0;

		// If we have a synapse that is not using this trigger, we just skip the processing
		if (prepos_dif!=0.0 || pospre_dif!=0.0 || ach_dif!=0.0) {

			Neuron * neuron = Connection->GetTarget();

			unsigned int LearningRuleId = this->GetLearningRuleIndex();

			for (int i = 0; i<neuron->GetInputNumberWithPostSynapticLearning(LearningRuleId); ++i) {
				Interconnection * interi = neuron->GetInputConnectionWithPostSynapticLearningAt(LearningRuleId, i);
				if (!interi->GetTriggerConnection()) {

					int LearningRuleIndex = neuron->IndexInputLearningConnections[0][LearningRuleId][i];

					// Apply synaptic activity decaying rule
					State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, true);

					// Apply ACh increment (1)
					State->incrementStateVariableAt(LearningRuleIndex, ESTDEState::ACH_INDEX, ach_dif);

					// Get current neuromodulator values
					float ach = interi->GetWeightChange_withPost()->State->GetStateVariableAt(LearningRuleIndex, ESTDEState::ACH_INDEX);
					// float ach = 1.0;
					ach = (ach>1.0) ? 1.0 : ach;
					// ach = 1.0 - ach;

					// Update synaptic weight
					float wdif;
					wdif =  prepos_dif * State->GetStateVariableAt(LearningRuleIndex, ESTDEState::PREPOS_INDEX);
					wdif += pospre_dif * State->GetStateVariableAt(LearningRuleIndex, ESTDEState::POSPRE_INDEX);
					wdif *= ach;
					interi->IncrementWeight(wdif);
				}
			}
		}

	}
}


void ESTDEWeightChange::ApplyPostSynapticSpike(Neuron * neuron, double SpikeTime){
  // If we have a synapse that is not using pre-pos or pos events, we just skip the processing
  unsigned int LearningRuleId = this->GetLearningRuleIndex();

	for (int i = 0; i<neuron->GetInputNumberWithPostSynapticLearning(LearningRuleId); ++i){
		Interconnection * interi = neuron->GetInputConnectionWithPostSynapticLearningAt(LearningRuleId, i);

		int LearningRuleIndex = neuron->IndexInputLearningConnections[0][LearningRuleId][i];
		// Apply synaptic activity decaying rule
		State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, true);

		// Get current neuromodulator values
		float ach = interi->GetWeightChange_withPost()->State->GetStateVariableAt(LearningRuleIndex, ESTDEState::ACH_INDEX);
		// float ach = 1.0;
		ach = (ach>1.0) ? 1.0 : ach;
		// ach = 1.0 - ach;

		// Apply postsynaptic spike
		State->ApplyPostsynapticSpike(LearningRuleIndex);

		// Update synaptic weight
		// float wdif = ach * this->prepos_dif * State->GetPresynapticActivity(LearningRuleIndex);
		float wdif = this->prepos_dif * State->GetPresynapticActivity(LearningRuleIndex) + this->pos_dif;
		interi->IncrementWeight(wdif);
	}

}


ModelDescription ESTDEWeightChange::ParseLearningRule(FILE * fh)  {
}


void ESTDEWeightChange::SetParameters(std::map<std::string, boost::any> param_map) {

	// Search for the parameters in the dictionary
	std::map<std::string, boost::any>::iterator it;

  it = param_map.find("pre_dif");
	if (it != param_map.end()) {
		this->pre_dif = boost::any_cast<float>(it->second);;
		param_map.erase(it);
	}

	it = param_map.find("prepos_dif");
	if (it != param_map.end()) {
		this->prepos_dif = boost::any_cast<float>(it->second);;
		param_map.erase(it);
	}

	it = param_map.find("tau_pre");
	if (it!=param_map.end()) {
		float v = boost::any_cast<float>(it->second);
		if (v<=0) {
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_STDP_WEIGHT_CHANGE_TAUS,
								 REPAIR_LEARNING_RULE_VALUES);
		}
		this->tau_pre = v;
		param_map.erase(it);
	}

	it = param_map.find("pospre_dif");
	if (it != param_map.end()) {
		this->pospre_dif = boost::any_cast<float>(it->second);;
		param_map.erase(it);
	}

	it=param_map.find("tau_pos");
	if (it!=param_map.end()) {
		float v = boost::any_cast<float>(it->second);
		if (v<=0) {
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_STDP_WEIGHT_CHANGE_TAUS,
								 REPAIR_LEARNING_RULE_VALUES);
		}
		this->tau_pos = v;
		param_map.erase(it);
	}

	it=param_map.find("tau_eli");
	if (it!=param_map.end()) {
		float v = boost::any_cast<float>(it->second);
		if (v<=0) {
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_STDP_WEIGHT_CHANGE_TAUS,
								 REPAIR_LEARNING_RULE_VALUES);
		}
		this->tau_eli = v;
		param_map.erase(it);
	}

	it=param_map.find("tau_ach");
	if (it!=param_map.end()) {
		float v = boost::any_cast<float>(it->second);
		if (v<=0) {
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_STDP_WEIGHT_CHANGE_TAUS,
								 REPAIR_LEARNING_RULE_VALUES);
		}
		this->tau_ach = v;
		param_map.erase(it);
	}

	it=param_map.find("prepos_tri_dif");
	if (it!=param_map.end()) {
		auto list = boost::any_cast< vector<float> >(it->second);
		for (std::size_t i=0; i<list.size(); ++i) {
			prepos_tri_dif[i] = list[i];
		}
		param_map.erase(it);
	}

	it=param_map.find("pospre_tri_dif");
	if (it!=param_map.end()) {
		auto list = boost::any_cast< vector<float> >(it->second);
		for (std::size_t i=0; i<list.size(); ++i) {
			pospre_tri_dif[i] = list[i];
		}
		param_map.erase(it);
	}

	it=param_map.find("ach_tri_dif");
	if (it!=param_map.end()) {
		auto list = boost::any_cast< vector<float> >(it->second);
		for (std::size_t i=0; i<list.size(); ++i) {
			ach_tri_dif[i] = list[i];
		}
		param_map.erase(it);
	}

  it = param_map.find("pos_dif");
	if (it != param_map.end()) {
		this->pos_dif = boost::any_cast<float>(it->second);;
		param_map.erase(it);
	}

	WithPostSynaptic::SetParameters(param_map);
}


ostream & ESTDEWeightChange::PrintInfo(ostream & out){
	out << "- ESTDE Learning Rule: " << endl;
	out << "\t prepos_dif:" << this->prepos_dif << endl;
	out << "\t tau_pre:" << this->tau_pre << endl;
	out << "\t pospre_dif:" << this->pospre_dif << endl;
	out << "\t tau_pos:" << this->tau_pos << endl;
	out << "\t tau_eli:" << this->tau_eli << endl;
	out << "\t tau_ach:" << this->tau_ach << endl;
	return out;
}

// float ESTDEWeightChange::GetMaxWeightChangeLTP() const{
// 	return this->MaxChangeLTP;
// }
//
// void ESTDEWeightChange::SetMaxWeightChangeLTP(float NewMaxChange){
// 	this->MaxChangeLTP = NewMaxChange;
// }
//
// float ESTDEWeightChange::GetMaxWeightChangeLTD() const{
// 	return this->MaxChangeLTD;
// }
//
// void ESTDEWeightChange::SetMaxWeightChangeLTD(float NewMaxChange){
// 	this->MaxChangeLTD = NewMaxChange;
// }

LearningRule* ESTDEWeightChange::CreateLearningRule(ModelDescription lrDescription){
	ESTDEWeightChange * lrule = new ESTDEWeightChange();
	lrule->SetParameters(lrDescription.param_map);
	return lrule;
}

std::map<std::string,boost::any> ESTDEWeightChange::GetParameters(){
	std::map<std::string,boost::any> newMap = WithPostSynaptic::GetParameters();
  	newMap["pre_dif"] = boost::any(this->pre_dif);
	newMap["prepos_dif"] = boost::any(this->prepos_dif);
	newMap["tau_pre"] = boost::any(this->tau_pre);
  	newMap["pos_dif"] = boost::any(this->pos_dif);
	newMap["pospre_dif"] = boost::any(this->pospre_dif);
	newMap["tau_pos"] = boost::any(this->tau_pos);
	newMap["tau_eli"] = boost::any(this->tau_eli);
	newMap["tau_ach"] = boost::any(this->tau_ach);

	// std::vector<float> floats;
	// floats = {0.01, -0.01, 0.0};
	// newMap["prepos_tri_dif"] = boost::any(std::string(floats.begin(), floats.end()));
	// floats = {0.01, -0.01, 0.0};
	// newMap["pospre_tri_dif"] = boost::any(std::string(floats.begin(), floats.end()));
	// floats = {0.0, 0.0, 1.0};
	// newMap["ach_tri_dif"] = boost::any(std::string(floats.begin(), floats.end()));

	return newMap;
}

std::map<std::string,boost::any> ESTDEWeightChange::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = WithPostSynaptic::GetDefaultParameters();
  	newMap["pre_dif"] = boost::any(0.0f);
  	newMap["pos_dif"] = boost::any(0.0f);
	newMap["prepos_dif"] = boost::any(0.01f);
	newMap["tau_pre"] = boost::any(0.032f);
	newMap["pospre_dif"] = boost::any(-0.02f);
	newMap["tau_pos"] = boost::any(0.032f);
	newMap["tau_eli"] = boost::any(0.1f);
	newMap["tau_ach"] = boost::any(0.5f);

	// std::vector<float> floats;
	// floats = {0.01, -0.01, 0.0};
	// newMap["prepos_tri_dif"] = boost::any(std::string(floats.begin(), floats.end()));
	// floats = {0.01, -0.01, 0.0};
	// newMap["pospre_tri_dif"] = boost::any(std::string(floats.begin(), floats.end()));
	// floats = {0.0, 0.0, 1.0};
	// newMap["ach_tri_dif"] = boost::any(std::string(floats.begin(), floats.end()));

	return newMap;
}
