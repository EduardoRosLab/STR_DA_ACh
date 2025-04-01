/***************************************************************************
 *                           NormalizedSTDPWeightChange.cpp                          *
 *                           -------------------                           *
 * copyright            : (C) 2023 by Álvaro GR                    *
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

#include "../../include/learning_rules/NormalizedSTDPWeightChange.h"

#include "../../include/learning_rules/NormalizedSTDPState.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/Utils.h"

#include "../../include/neuron_model/NeuronState.h"


NormalizedSTDPWeightChange::NormalizedSTDPWeightChange():WithPostSynaptic(){
	// Set the default values for the learning rule parameters
	this->SetParameters(NormalizedSTDPWeightChange::GetDefaultParameters());
}

NormalizedSTDPWeightChange::~NormalizedSTDPWeightChange(){

}


void NormalizedSTDPWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons){
	this->State=(ConnectionState *) new NormalizedSTDPState(NumberOfSynapses, this->tauPrePos, this->tauPosPre, this->tauEligibility);
}


void NormalizedSTDPWeightChange::ApplyPreSynapticSpike(Interconnection * Connection, double SpikeTime){
	unsigned int LearningRuleIndex = Connection->GetLearningRuleIndex_withPost();
	if (this->tauEligibility<0) {
		// Apply synaptic activity decaying rule
		State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

		// Apply presynaptic spike
		State->ApplyPresynapticSpike(LearningRuleIndex);

		// Apply weight change	
		float current_weight = Connection->GetWeight();
		float pospre = this->MaxChangePosPre * State->GetPostsynapticActivity(LearningRuleIndex);
		float WeightChange = pospre * (this->a*current_weight + (1.0f-this->a));

		Connection->IncrementWeight(WeightChange - this->preLTD);
	}
	else {
		double SumWeights = 0.0;
		double DesiredSumWeights = this->wSum;  // or any constant value you prefer

		// Normal spike, just update eligibility traces
		if (!Connection->GetTriggerConnection()) {
			State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);
			State->ApplyPresynapticSpike(LearningRuleIndex);
		}
		// Trigger spike, use eligibility traces to update weights
		else {
			int conn_type = Connection->GetType();
			int sign = (conn_type==0) ? 1 : -1;
			Neuron * neuron = Connection->GetTarget();

			unsigned int LearningRuleId = this->GetLearningRuleIndex();

			for (int i = 0; i<neuron->GetInputNumberWithPostSynapticLearning(LearningRuleId); ++i) {
				Interconnection* interi = neuron->GetInputConnectionWithPostSynapticLearningAt(LearningRuleId, i);
				if (!interi->GetTriggerConnection()) {

					int LearningRuleIndex = neuron->IndexInputLearningConnections[0][LearningRuleId][i];

					// Apply synaptic activity decaying rule
					State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, true);

					// Update synaptic weight
					float wdif;
					wdif =  this->MaxChangePrePos * State->GetStateVariableAt(LearningRuleIndex, NormalizedSTDPState::PREPOS);
					wdif += this->MaxChangePosPre * State->GetStateVariableAt(LearningRuleIndex, NormalizedSTDPState::POSPRE);

					interi->IncrementWeight(wdif*sign);

					// Accumulate the sum of weights
					SumWeights += interi->GetWeight();
				}
			}

			if (this->normalize) {
				// Calculate the normalization factor
				double NormalizationFactor = DesiredSumWeights / SumWeights;

				// Normalize the weights
				for (int i = 0; i < neuron->GetInputNumberWithPostSynapticLearning(LearningRuleId); ++i){
					Interconnection* interi = neuron->GetInputConnectionWithPostSynapticLearningAt(LearningRuleId, i);
					if (!interi->GetTriggerConnection()) {
						double NewWeight = interi->GetWeight() * NormalizationFactor;
						interi->SetWeight(NewWeight);
					}
				}
			}
		}
	}

	return;
}

void NormalizedSTDPWeightChange::ApplyPostSynapticSpike(Neuron * neuron, double SpikeTime){
    unsigned int LearningRuleId = this->GetLearningRuleIndex();

	if (this->tauEligibility<0) {
		double SumWeights = 0.0;
		double DesiredSumWeights = this->wSum;  // or any constant value you prefer

		unsigned int NumberOfSynapses = neuron->GetInputNumberWithPostSynapticLearning(LearningRuleId);
		for (int i = 0; i<NumberOfSynapses; ++i){
			Interconnection * interi = neuron->GetInputConnectionWithPostSynapticLearningAt(LearningRuleId, i);

			unsigned int LearningRuleIndex = neuron->IndexInputLearningConnections[0][LearningRuleId][i];

			// Apply synaptic activity decaying rule
			State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, true);

			// Apply postsynaptic spike
			State->ApplyPostsynapticSpike(LearningRuleIndex);

			// Update synaptic weight
			float current_weight = interi->GetWeight();
			float prepos = this->MaxChangePrePos * State->GetPresynapticActivity(LearningRuleIndex);
			float WeightChange = prepos * (1.0f - this->a*current_weight);

			interi->IncrementWeight(WeightChange);

			// Accumulate the sum of weights
			SumWeights += interi->GetWeight();
		}

		if (this->normalize) {
			// Calculate the normalization factor
			double NormalizationFactor = DesiredSumWeights / SumWeights;

			// Normalize the weights
			for (int i = 0; i < neuron->GetInputNumberWithPostSynapticLearning(LearningRuleId); ++i){
				Interconnection * interi = neuron->GetInputConnectionWithPostSynapticLearningAt(LearningRuleId, i);
				double NewWeight = interi->GetWeight() * NormalizationFactor;
				interi->SetWeight(NewWeight);
			}
		}
	}
	else {
		unsigned int NumberOfSynapses = neuron->GetInputNumberWithPostSynapticLearning(LearningRuleId);
		for (int i = 0; i<NumberOfSynapses; ++i) {
			Interconnection * interi = neuron->GetInputConnectionWithPostSynapticLearningAt(LearningRuleId, i);
			unsigned int LearningRuleIndex = neuron->IndexInputLearningConnections[0][LearningRuleId][i];
			State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, true);
			State->ApplyPostsynapticSpike(LearningRuleIndex);
		}
	}
}


ModelDescription NormalizedSTDPWeightChange::ParseLearningRule(FILE * fh)  {
	ModelDescription lrule;

	float ltauPrePos, lMaxChangePrePos, ltauPosPre, lMaxChangePosPre, ltauEligibility;
	if(fscanf(fh,"%f",&lMaxChangePrePos)!=1 ||
	   fscanf(fh,"%f",&ltauPrePos)!=1 ||
	   fscanf(fh,"%f",&lMaxChangePosPre)!=1 ||
	   fscanf(fh,"%f",&ltauPosPre)!=1 ||
	   fscanf(fh,"%f",&ltauEligibility)!=1){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_LEARNING_RULE_LOAD, REPAIR_STDP_WEIGHT_CHANGE_LOAD);
	}
	if (ltauPrePos <= 0 || ltauPosPre <= 0){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_STDP_WEIGHT_CHANGE_TAUS, REPAIR_LEARNING_RULE_VALUES);
	}

	lrule.model_name = NormalizedSTDPWeightChange::GetName();
	lrule.param_map["max_PrePos"] = boost::any(lMaxChangePrePos);
	lrule.param_map["tau_PrePos"] = boost::any(ltauPrePos);
	lrule.param_map["max_PosPre"] = boost::any(lMaxChangePosPre);
	lrule.param_map["tau_PosPre"] = boost::any(ltauPosPre);
	lrule.param_map["tau_Eligibility"] = boost::any(ltauEligibility);

	return lrule;
}

void NormalizedSTDPWeightChange::SetParameters(std::map<std::string, boost::any> param_map) {

	// Search for the parameters in the dictionary
	std::map<std::string, boost::any>::iterator it = param_map.find("max_PrePos");
	if (it != param_map.end()){
		this->MaxChangePrePos = boost::any_cast<float>(it->second);;
		param_map.erase(it);
	}

	it=param_map.find("tau_PrePos");
	if (it!=param_map.end()){
		float newtauPrePos = boost::any_cast<float>(it->second);
		if (newtauPrePos<=0) {
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_STDP_WEIGHT_CHANGE_TAUS,
								 REPAIR_LEARNING_RULE_VALUES);
		}
		this->tauPrePos = newtauPrePos;
		param_map.erase(it);
	}

	it = param_map.find("max_PosPre");
	if (it != param_map.end()){
		this->MaxChangePosPre = boost::any_cast<float>(it->second);;
		param_map.erase(it);
	}

	it=param_map.find("tau_PosPre");
	if (it!=param_map.end()){
		float newtauPosPre = boost::any_cast<float>(it->second);
		if (newtauPosPre<=0) {
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_STDP_WEIGHT_CHANGE_TAUS,
								 REPAIR_LEARNING_RULE_VALUES);
		}
		this->tauPosPre = newtauPosPre;
		param_map.erase(it);
	}

	it=param_map.find("tau_Eligibility");
	if (it!=param_map.end()){
		float newtauEligibility = boost::any_cast<float>(it->second);
		this->tauEligibility = newtauEligibility;
		param_map.erase(it);
	}

	it=param_map.find("a");
	if (it!=param_map.end()){
		float new_a = boost::any_cast<float>(it->second);
		this->a = new_a;
		param_map.erase(it);
	}

	it=param_map.find("normalize");
	if (it!=param_map.end()){
		int new_normalize = boost::any_cast<int>(it->second);
		this->normalize = new_normalize;
		param_map.erase(it);
	}

	it=param_map.find("preLTD");
	if (it!=param_map.end()){
		float new_preLTD = boost::any_cast<float>(it->second);
		this->preLTD = new_preLTD;
		param_map.erase(it);
	}

	it=param_map.find("wSum");
	if (it!=param_map.end()){
		float new_wSum = boost::any_cast<float>(it->second);
		this->wSum = new_wSum;
		param_map.erase(it);
	}

	WithPostSynaptic::SetParameters(param_map);
}


ostream & NormalizedSTDPWeightChange::PrintInfo(ostream & out){
	out << "- STDP Learning Rule: " << endl;
	out << "\t max_PrePos:" << this->MaxChangePrePos << endl;
	out << "\t tau_PrePos:" << this->tauPrePos << endl;
	out << "\t max_PosPre:" << this->MaxChangePosPre << endl;
	out << "\t tau_PosPre:" << this->tauPosPre << endl;
	out << "\t tau_Eligibility:" << this->tauEligibility << endl;
	out << "\t a:" << this->a << endl;
	out << "\t normalize:" << this->normalize << endl;
	out << "\t preLTD:" << this->preLTD << endl;
	out << "\t wSum:" << this->wSum << endl;
	return out;
}

float NormalizedSTDPWeightChange::GetMaxWeightChangeLTP() const{
	return this->MaxChangePrePos;
}

void NormalizedSTDPWeightChange::SetMaxWeightChangeLTP(float NewMaxChange){
	this->MaxChangePrePos = NewMaxChange;
}

float NormalizedSTDPWeightChange::GetMaxWeightChangeLTD() const{
	return this->MaxChangePosPre;
}

void NormalizedSTDPWeightChange::SetMaxWeightChangeLTD(float NewMaxChange){
	this->MaxChangePosPre = NewMaxChange;
}

LearningRule* NormalizedSTDPWeightChange::CreateLearningRule(ModelDescription lrDescription){
	NormalizedSTDPWeightChange * lrule = new NormalizedSTDPWeightChange();
	lrule->SetParameters(lrDescription.param_map);
	return lrule;
}

std::map<std::string,boost::any> NormalizedSTDPWeightChange::GetParameters(){
	std::map<std::string,boost::any> newMap = WithPostSynaptic::GetParameters();
	newMap["max_PrePos"] = boost::any(this->MaxChangePrePos);
	newMap["tau_PrePos"] = boost::any(this->tauPrePos);
	newMap["max_PosPre"] = boost::any(this->MaxChangePosPre);
	newMap["tau_PosPre"] = boost::any(this->tauPosPre);
	newMap["tau_Eligibility"] = boost::any(this->tauEligibility);
	newMap["a"] = boost::any(this->a);
	newMap["normalize"] = boost::any(this->normalize);
	newMap["preLTD"] = boost::any(this->preLTD);
	newMap["wSum"] = boost::any(this->wSum);
	return newMap;
}

std::map<std::string,boost::any> NormalizedSTDPWeightChange::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = WithPostSynaptic::GetDefaultParameters();
	newMap["max_PrePos"] = boost::any(0.016f);
	newMap["tau_PrePos"] = boost::any(0.100f);
	newMap["max_PosPre"] = boost::any(0.033f);
	newMap["tau_PosPre"] = boost::any(0.100f);
	newMap["tau_Eligibility"] = boost::any(-1.000f);
	newMap["a"] = boost::any(0.0f);
	newMap["normalize"] = boost::any(0);
	newMap["preLTD"] = boost::any(0.0f);
	newMap["wSum"] = boost::any(1.0f);
	return newMap;
}
