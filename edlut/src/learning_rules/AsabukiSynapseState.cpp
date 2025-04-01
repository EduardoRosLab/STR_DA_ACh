/***************************************************************************
 *                           AsabukiSynapseState.cpp                       *
 *                           -------------------                           *
 * copyright            : (C) 2022 by Álvaro González-Redondo              *
 * email                : jgarrido@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/learning_rules/AsabukiSynapseState.h"

#include "../../include/simulation/ExponentialTable.h"

#include <cmath>
#include <stdio.h>
#include <float.h>

AsabukiSynapseState::AsabukiSynapseState(int NumSynapses, int new_is_lateral, float new_tau_p, float new_tau_d, float new_Cp, float new_Cd)
	: ConnectionState(NumSynapses, 4), is_lateral(new_is_lateral), tau_p(new_tau_p), tau_d(new_tau_d), Cp(new_Cp), Cd(new_Cd) {
		inv_tau_p = 1.0f / new_tau_p;
		inv_tau_d = 1.0f / new_tau_d;
	}

AsabukiSynapseState::~AsabukiSynapseState()
{
}

unsigned int AsabukiSynapseState::GetNumberOfPrintableValues()
{
	return ConnectionState::GetNumberOfPrintableValues() + 4;
}

double AsabukiSynapseState::GetPrintableValuesAt(unsigned int index, unsigned int position)
{
	if (position < ConnectionState::GetNumberOfPrintableValues())
	{
		return ConnectionState::GetStateVariableAt(index, position);
	}
	else
		return -1;
}

// float AsabukiSynapseState::GetPresynapticActivity(unsigned int index){
//	return this->GetStateVariableAt(index, 0);
// }

// float AsabukiSynapseState::GetPostsynapticActivity(unsigned int index){
//	return this->GetStateVariableAt(index, 1);
// }

// void AsabukiSynapseState::SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post){
//	float PreActivity = this->GetPresynapticActivity(index);
//	float PostActivity = this->GetPostsynapticActivity(index);
//
//	float ElapsedTime=(float)(NewTime - this->GetLastUpdateTime(index));
//
//	//// Accumulate activity since the last update time
//	PreActivity *= exp(-ElapsedTime*this->inv_LTPTau);
//	PostActivity *= exp(-ElapsedTime*this->inv_LTDTau);
//
//	// Store the activity in state variables
//	//this->SetStateVariableAt(index, 0, PreActivity);
//	//this->SetStateVariableAt(index, 1, PostActivity);
//	this->SetStateVariableAt(index, 0, PreActivity, PostActivity);
//
//	this->SetLastUpdateTime(index, NewTime);
// }

void AsabukiSynapseState::SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post){
	float ElapsedTime=(float)(NewTime - this->GetLastUpdateTime(index));

    //Accumulate activity since the last update time
	if (this->is_lateral) {
		this->multiplyStateVariableAt(index,0,ExponentialTable::GetResult(-ElapsedTime*this->inv_tau_p));
		this->multiplyStateVariableAt(index,1,ExponentialTable::GetResult(-ElapsedTime*this->inv_tau_d));
		this->multiplyStateVariableAt(index,2,ExponentialTable::GetResult(-ElapsedTime*this->inv_tau_p));
		this->multiplyStateVariableAt(index,3,ExponentialTable::GetResult(-ElapsedTime*this->inv_tau_d));
	}

	this->SetLastUpdateTime(index, NewTime);
}

void AsabukiSynapseState::ApplyPresynapticSpike(unsigned int index)
{
	// Increment the activity in the state variable
	if (this->is_lateral) {
		this->incrementStateVariableAt(index, 0, this->Cp);
		this->incrementStateVariableAt(index, 1, this->Cd);
	}
}

void AsabukiSynapseState::ApplyPostsynapticSpike(unsigned int index)
{
	// Increment the activity in the state variable
	if (this->is_lateral) {
		this->incrementStateVariableAt(index, 2, this->Cp);
		this->incrementStateVariableAt(index, 3, this->Cd);
	}
}