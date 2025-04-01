/***************************************************************************
 *                           NormalizedSTDPState.cpp                                 *
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

#include "../../include/learning_rules/NormalizedSTDPState.h"

#include "../../include/simulation/ExponentialTable.h"

#include <cmath>
#include <stdio.h>
#include <float.h>

NormalizedSTDPState::NormalizedSTDPState(
	int NumSynapses, 
	float NewLTPValue, 
	float NewLTDValue,
	float NewEligibilityValue
	): 
	ConnectionState(NumSynapses, N_STATES),
	LTPTau(NewLTPValue),
	LTDTau(NewLTDValue),
	eligibilityTau(NewEligibilityValue)
{
	inv_LTPTau = 1.0f/NewLTPValue;
	inv_LTDTau = 1.0f/NewLTDValue;
	inv_eligibilityTau = 1.0f/NewEligibilityValue;
}

NormalizedSTDPState::~NormalizedSTDPState() {
}

unsigned int NormalizedSTDPState::GetNumberOfPrintableValues(){
	return ConnectionState::GetNumberOfPrintableValues()+2;
}

double NormalizedSTDPState::GetPrintableValuesAt(unsigned int index, unsigned int position){
	if (position<ConnectionState::GetNumberOfPrintableValues()){
		return ConnectionState::GetStateVariableAt(index, position);
	} else if (position==ConnectionState::GetNumberOfPrintableValues()) {
		return this->LTPTau;
	} else if (position==ConnectionState::GetNumberOfPrintableValues()+1) {
		return this->LTDTau;
	} else return -1;
}


void NormalizedSTDPState::SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post){
	float ElapsedTime=(float)(NewTime - this->GetLastUpdateTime(index));

    //Accumulate activity since the last update time
	this->multiplyStateVariableAt(index, PRE, ExponentialTable::GetResult(-ElapsedTime*this->inv_LTPTau));
	this->multiplyStateVariableAt(index, POS, ExponentialTable::GetResult(-ElapsedTime*this->inv_LTDTau));
	this->multiplyStateVariableAt(index, PREPOS, ExponentialTable::GetResult(-ElapsedTime*this->inv_eligibilityTau));
	this->multiplyStateVariableAt(index, POSPRE, ExponentialTable::GetResult(-ElapsedTime*this->inv_eligibilityTau));
	
	this->SetLastUpdateTime(index, NewTime);
}


void NormalizedSTDPState::ApplyPresynapticSpike(unsigned int index){
	// Increment the activity in the state variable
	this->incrementStateVariableAt(index, PRE, 1.0f);
	float pos = this->GetStateVariableAt(index, POS);
	this->incrementStateVariableAt(index, POSPRE, pos);
}

void NormalizedSTDPState::ApplyPostsynapticSpike(unsigned int index){
	// Increment the activity in the state variable
	this->incrementStateVariableAt(index, POS, 1.0f); 
	float pre = this->GetStateVariableAt(index, PRE);
	this->incrementStateVariableAt(index, PREPOS, pre);
}

