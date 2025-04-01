/***************************************************************************
 *                           ESTDEState.cpp                                 *
 *                           -------------------                           *
 * copyright            : (C) 2022 Álvaro González-Redondo                  *
 * email                : alvarogr@ugr.es                 *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/learning_rules/ESTDEState.h"

#include "../../include/simulation/ExponentialTable.h"

#include <cmath>
#include <stdio.h>
#include <float.h>
#include <iostream>

ESTDEState::ESTDEState(
	int n_syn,
	float new_tau_pre,
	float new_tau_pos,
	float new_tau_eli,
	float new_tau_ach
):
ConnectionState(n_syn, N_STATES),
tau_pre(new_tau_pre),
tau_pos(new_tau_pos),
tau_eli(new_tau_eli),
tau_ach(new_tau_ach)
{
	inv_tau_pre = 1.0f/new_tau_pre;
	inv_tau_pos = 1.0f/new_tau_pos;
	inv_tau_eli = 1.0f/new_tau_eli;
	inv_tau_ach = 1.0f/new_tau_ach;
}

ESTDEState::~ESTDEState() {
}

unsigned int ESTDEState::GetNumberOfPrintableValues() {
	return ConnectionState::GetNumberOfPrintableValues() + 2;
}

double ESTDEState::GetPrintableValuesAt(unsigned int index, unsigned int position) {
	if (position<ConnectionState::GetNumberOfPrintableValues()) {
		return ConnectionState::GetStateVariableAt(index, position);
	} else if (position == ConnectionState::GetNumberOfPrintableValues()) {
		return this->tau_pre;
	} else if (position == ConnectionState::GetNumberOfPrintableValues()+1) {
		return this->tau_pos;
	} else return -1;
}


void ESTDEState::SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post) {
	float ElapsedTime=(float)(NewTime - this->GetLastUpdateTime(index));

  //Accumulate activity since the last update time
	this->multiplyStateVariableAt(index, PRE_INDEX, ExponentialTable::GetResult(-ElapsedTime*this->inv_tau_pre));
	this->multiplyStateVariableAt(index, POS_INDEX, ExponentialTable::GetResult(-ElapsedTime*this->inv_tau_pos));
	this->multiplyStateVariableAt(index, PREPOS_INDEX, ExponentialTable::GetResult(-ElapsedTime*this->inv_tau_eli));
	this->multiplyStateVariableAt(index, POSPRE_INDEX, ExponentialTable::GetResult(-ElapsedTime*this->inv_tau_eli));
	this->multiplyStateVariableAt(index, ACH_INDEX, ExponentialTable::GetResult(-ElapsedTime*this->inv_tau_ach));

	// std::cout << "Hola" << std::endl;

	this->SetLastUpdateTime(index, NewTime);
}


void ESTDEState::ApplyPresynapticSpike(unsigned int index) {
	// Increment the activity in the state variable
	this->incrementStateVariableAt(index, PRE_INDEX, 1.0f);
	float pos = this->GetStateVariableAt(index, POS_INDEX);
	this->incrementStateVariableAt(index, POSPRE_INDEX, pos);
}

void ESTDEState::ApplyPostsynapticSpike(unsigned int index) {
	// Increment the activity in the state variable
	this->incrementStateVariableAt(index, POS_INDEX, 1.0f);
	float pre = this->GetStateVariableAt(index, PRE_INDEX);
	this->incrementStateVariableAt(index, PREPOS_INDEX, pre);
}

void ESTDEState::SetWeight(unsigned int index, float weight, float max_weight){
}
