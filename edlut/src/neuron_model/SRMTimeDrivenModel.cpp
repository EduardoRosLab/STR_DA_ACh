/***************************************************************************
 *                           SRMTimeDrivenModel.cpp                        *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido and Francisco Naveros  *
 * email                : jgarrido@atc.ugr.es, fnaveros@ugr.es             *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <iostream>
#include <cmath>
#include <string>

#include "../../include/neuron_model/SRMTimeDrivenModel.h"
#include "../../include/neuron_model/VectorSRMState.h"
#include "../../include/neuron_model/VectorNeuronState.h"

#include "../../include/spike/EDLUTFileException.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/PropagatedSpike.h"

#include "../../include/simulation/Utils.h"
#include "../../include/simulation/RandomGenerator.h"

#include "../../include/openmp/openmp.h"


using namespace std;

void SRMTimeDrivenModel::LoadNeuronModel(string ConfigFile) {
	FILE *fh;
	long Currentline = 0L;

	fh=fopen(ConfigFile.c_str(),"rt");
	if(!fh){
		// Error: Neuron model file doesn't exist
		throw EDLUTFileException(TASK_SRM_TIME_DRIVEN_MODEL_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, Currentline, ConfigFile.c_str());
	}

	Currentline=1L;
	skip_comments(fh,Currentline);
	if (!(fscanf(fh, "%d", &this->NumberOfChannels) == 1) || this->NumberOfChannels <= 0){
		throw EDLUTFileException(TASK_SRM_TIME_DRIVEN_MODEL_LOAD, ERROR_SRM_TIME_DRIVEN_MODEL_NUMBER_OF_CHANNELS, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str());
	}

	this->tau = (float *) new float [this->NumberOfChannels];
	this->W = (float *) new float [this->NumberOfChannels];

	for (unsigned int i=0; i<this->NumberOfChannels; ++i){
		skip_comments(fh,Currentline);
		if(!(fscanf(fh,"%f",&this->tau[i])==1)){
			throw EDLUTFileException(TASK_SRM_TIME_DRIVEN_MODEL_LOAD, ERROR_SRM_TIME_DRIVEN_MODEL_TAUS, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str());
		}
	}

	skip_comments(fh,Currentline);

	if(!(fscanf(fh,"%f",&this->vr)==1)){
		throw EDLUTFileException(TASK_SRM_TIME_DRIVEN_MODEL_LOAD, ERROR_SRM_TIME_DRIVEN_MODEL_VR, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str());
	}

	for (unsigned int i=0; i<this->NumberOfChannels; ++i){
		skip_comments(fh,Currentline);
		if(!(fscanf(fh,"%f",&this->W[i])==1)){
			throw EDLUTFileException(TASK_SRM_TIME_DRIVEN_MODEL_LOAD, ERROR_SRM_TIME_DRIVEN_MODEL_W, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str());
		}
	}

	skip_comments(fh,Currentline);

	if(!(fscanf(fh,"%f",&this->r0)==1)){
		throw EDLUTFileException(TASK_SRM_TIME_DRIVEN_MODEL_LOAD, ERROR_SRM_TIME_DRIVEN_MODEL_R0, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str());
	}

	skip_comments(fh,Currentline);

	if(!(fscanf(fh,"%f",&this->v0)==1)){
		throw EDLUTFileException(TASK_SRM_TIME_DRIVEN_MODEL_LOAD, ERROR_SRM_TIME_DRIVEN_MODEL_V0, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str());
	}

	skip_comments(fh,Currentline);

	if(!(fscanf(fh,"%f",&this->vf)==1)){
		throw EDLUTFileException(TASK_SRM_TIME_DRIVEN_MODEL_LOAD, ERROR_SRM_TIME_DRIVEN_MODEL_VF, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str());
	}

	skip_comments(fh,Currentline);

	if(!(fscanf(fh,"%f",&this->tauabs)==1)){
		throw EDLUTFileException(TASK_SRM_TIME_DRIVEN_MODEL_LOAD, ERROR_SRM_TIME_DRIVEN_MODEL_TAUABS, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str());
	}

	skip_comments(fh,Currentline);

	if(!(fscanf(fh,"%f",&this->taurel)==1)){
		throw EDLUTFileException(TASK_SRM_TIME_DRIVEN_MODEL_LOAD, ERROR_SRM_TIME_DRIVEN_MODEL_TAUREL, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str());
	}

	// Initialize the neuron state
	this->State = (VectorSRMState *) new VectorSRMState(5,this->NumberOfChannels, true);

	//TIME DRIVEN STEP
	this->integrationMethod = LoadIntegrationMethod::loadIntegrationMethod((TimeDrivenNeuronModel *)this, this->GetModelID(), fh, &Currentline, 0, 0, 0);


	fclose(fh);
}

void SRMTimeDrivenModel::SynapsisEffect(int index, Interconnection * InputConnection){
	((VectorSRMState *) this->GetVectorNeuronState())->AddActivity(index, InputConnection);
}

float SRMTimeDrivenModel::PotentialIncrement(int index, VectorSRMState * State){
	float Increment = 0;

	for (unsigned int i=0; i<this->NumberOfChannels; ++i){
		VectorBufferedState::Iterator itEnd = State->End();
		
		for (VectorBufferedState::Iterator it=State->Begin(index,i); it!=itEnd; ++it){
			float TimeDifference = it.GetSpikeTime();
			float Weight = it.GetConnection()->GetWeight();

			float EPSPMax = sqrt(this->tau[i]/2)*ExponentialTable::GetResult(-0.5);

			float EPSP = sqrt(TimeDifference)*ExponentialTable::GetResult(-(TimeDifference/this->tau[i]))/EPSPMax;

			// Inhibitory channels must define negative W values
			Increment += Weight*this->W[i]*EPSP;
		}
	}

	return Increment;
}

bool SRMTimeDrivenModel::CheckSpikeAt(int index, VectorSRMState * State, double CurrentTime){
	double Probability = State->GetStateVariableAt(index,4);
	return (RandomGenerator::drand()<Probability);
}

//this neuron model is implemented in a second scale.
SRMTimeDrivenModel::SRMTimeDrivenModel(string NeuronTypeID, string NeuronModelID): TimeDrivenNeuronModel(NeuronTypeID, NeuronModelID, SecondScale), tau(0), vr(0), W(0), r0(0), v0(0), vf(0),
		tauabs(0), taurel(0) {

}

SRMTimeDrivenModel::~SRMTimeDrivenModel(){
	delete [] this->tau;
	this->tau = 0;
	delete [] this->W;
	this->W = 0;
}

void SRMTimeDrivenModel::LoadNeuronModel()  {

	this->LoadNeuronModel(this->GetModelID() + ".cfg");
}

VectorNeuronState * SRMTimeDrivenModel::InitializeState(){
	return ((VectorSRMState *) State);
}


InternalSpike * SRMTimeDrivenModel::ProcessInputSpike(Interconnection * inter, Neuron * target, double time){


	InternalSpike * ProducedSpike = 0;

	// Update Cell State
	if (this->UpdateState(target->GetIndex_VectorNeuronState(),time)){
		ProducedSpike = new InternalSpike(time,target);
	}

	// Add the effect of the input spike
	this->SynapsisEffect(target->GetIndex_VectorNeuronState(),inter);

	return ProducedSpike;
}


bool SRMTimeDrivenModel::UpdateState(int index, double CurrentTime){

	float * sqrt_tau = new float[this->NumberOfChannels];
	for (unsigned int i=0; i<this->NumberOfChannels; ++i){
		sqrt_tau[i]=sqrt(this->tau[i]/2)*ExponentialTable::GetResult(-0.5);
	}

	VectorSRMState * SRMstate=(VectorSRMState *)State;

	bool * internalSpike=State->getInternalSpike();
	int Size=State->GetSizeState();
	int i;
	double ElapsedTime;

	for (i=0; i<Size; i++){
		ElapsedTime = CurrentTime-State->GetLastUpdateTime(i);

		State->AddElapsedTime(i, ElapsedTime);

		///////////////////////////
		float Increment = 0;
		
		for (unsigned int j=0; j<this->NumberOfChannels; ++j){
			VectorBufferedState::Iterator itEnd = SRMstate->End();
			
			for (VectorBufferedState::Iterator it=SRMstate->Begin(i,j); it!=itEnd; ++it){
				float TimeDifference = it.GetSpikeTime();
				float Weight = it.GetConnection()->GetWeight();

				float EPSPMax = sqrt_tau[j];

				float EPSP = sqrt(TimeDifference)*ExponentialTable::GetResult(-(TimeDifference/this->tau[j]))/EPSPMax;

				// Inhibitory channels must define negative W values
				Increment += Weight*this->W[j]*EPSP;
			}
		}

		///////////////////////////


		float Potential = this->vr + Increment;
		State->SetStateVariableAt(i,1,Potential);

		float FiringRate;

		if((Potential-this->v0) > (10*this->vf)){
			FiringRate = this->r0*(Potential-this->v0)/this->vf;
		} else {
			float texp=ExponentialTable::GetResult((Potential-this->v0)/this->vf);
		    FiringRate =this->r0*log(1+texp);
		}

		//double texp = ExponentialTable::GetResult((Potential-this->v0)/this->vf);
		//double FiringRate = this->r0 * log(1+texp);
		State->SetStateVariableAt(i,2,FiringRate);

		double TimeSinceSpike = State->GetLastSpikeTime(i);
		float Aux = TimeSinceSpike-this->tauabs;
		float Refractoriness = 0;

		if (TimeSinceSpike>this->tauabs){
			Refractoriness = 1./(1.+(this->taurel*this->taurel)/(Aux*Aux));
		}
		State->SetStateVariableAt(i,3,Refractoriness);

		float Probability = (1 - ExponentialTable::GetResult(-FiringRate*Refractoriness*((float)ElapsedTime)));
		State->SetStateVariableAt(i,4,Probability);

		State->SetLastUpdateTime(i,CurrentTime);

		if (this->CheckSpikeAt(i,(VectorSRMState *) State, CurrentTime)){
			State->NewFiredSpike(i);
			internalSpike[i]=true;
		}else{
			internalSpike[i]=false;
		}

	}
	delete [] sqrt_tau;

	return false;
}





ostream & SRMTimeDrivenModel::PrintInfo(ostream & out) {
	out << "- SRM Time-Driven Model: " << this->GetModelID() << endl;

	out << "\tNumber of channels: " << this->NumberOfChannels << endl;

	out << "\tTau: ";
	for (unsigned int i=0; i<this->NumberOfChannels; ++i){
		out << "\t" << this->tau[i];
	}

	out << endl << "\tVresting: " << this->vr << endl;

	out << "\tWeight Scale: ";

	for (unsigned int i=0; i<this->NumberOfChannels; ++i){
		out << "\t" << this->W[i];
	}

	out << endl << "\tFiring Rate: " << this->r0 << "Hz\tVthreshold: " << this->v0 << "V" << endl;

	out << "\tGain Factor: " << this->vf << "\tAbsolute Refractory Period: " << this->tauabs << "s\tRelative Refractory Period: " << this->taurel << "s" << endl;

	return out;
}


void SRMTimeDrivenModel::InitializeStates(int N_neurons, int OpenMPQueueIndex){

	VectorSRMState * state = (VectorSRMState *) this->State;

	float initialization[] = {0.0,0.0,0.0,0.0,0.0};
	//Initialize the state variables
	state->InitializeSRMStates(N_neurons, initialization);

	for (int j=0; j<N_neurons; j++){
		// Initialize the amplitude of each buffer
		for (unsigned int i=0; i<this->NumberOfChannels; ++i){
			state->SetBufferAmplitude(j,i,8*this->tau[i]);
		}
	}

	this->integrationMethod->InitializeStates(N_neurons, initialization);

}


bool SRMTimeDrivenModel::CheckSynapseTypeNumber(int Type){
	if (Type<NumberOfChannels && Type >= 0){
		return true;
	}
	else{
		cout << "Neuron model " << this->GetTypeID() << ", " << this->GetModelID() << " does not support input synapses of type " << Type << ". Just defined " << NumberOfChannels << " synapses types." << endl;
		return false;
	}
}