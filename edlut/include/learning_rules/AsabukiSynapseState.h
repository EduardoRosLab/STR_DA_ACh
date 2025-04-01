/***************************************************************************
 *                           AsabukiSynapseState.h                         *
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

#ifndef ASABUKISYNAPSESTATE_H_
#define ASABUKISYNAPSESTATE_H_

#include "ConnectionState.h"

/*!
 * \file AsabukiSynapseState.h
 *
 * \author Álvaro González-Redondo
 * \date June 2022
 *
 * This file declares a class which abstracts the current state of a synaptic
 * connection with the learning rule of Asabuki et al. (2020).
 */

/*!
 * \class AsabukiSynapseState
 *
 * \brief Synaptic connection current state.
 *
 * This class abstracts the current state of a synaptic connection with the
 * learning rule of Asabuki et al. (2020).
 *
 * \author Álvaro González-Redondo
 * \date June 2022
 */

class AsabukiSynapseState : public ConnectionState
{

public:
	/*!
		* Is lateral connection?.
		*/
	int is_lateral;

	/*!
		* LTP time constant.
		*/
	float tau_p;
	float inv_tau_p;

	/*!
		* LTD time constant.
		*/
	float tau_d;
	float inv_tau_d;

	/*!
		* LTP strength.
		*/
	float Cp;

	/*!
		* LTD strength.
		*/
	float Cd;


	/*!
	 * \brief Default constructor with parameters.
	 *
	 * It generates a new state of a connection.
	 */
	AsabukiSynapseState(int NumSynapses, int new_is_lateral, float new_tau_p, float new_tau_d, float new_Cp, float new_Cd);

	/*!
	 * \brief Class destructor.
	 *
	 * It destroys an object of this class.
	 */
	virtual ~AsabukiSynapseState();

	/*!
	 * \brief It gets the value of the accumulated presynaptic activity.
	 *
	 * It gets the value of the accumulated presynaptic activity.
	 *
	 * \return The accumulated presynaptic activity.
	 */
	// float GetPresynapticActivity(unsigned int index);
	inline float GetPresynapticActivity(unsigned int index)
	{
		return this->GetStateVariableAt(index, 0) + this->GetStateVariableAt(index, 1);
	}

	/*!
	 * \brief It gets the value of the accumulated postsynaptic activity.
	 *
	 * It gets the value of the accumulated postsynaptic activity.
	 *
	 * \return The accumulated postsynaptic activity.
	 */
	// float GetPostsynapticActivity(unsigned int index);
	inline float GetPostsynapticActivity(unsigned int index)
	{
		return this->GetStateVariableAt(index, 2) + this->GetStateVariableAt(index, 3);
	}

	/*!
	 * \brief It gets the number of variables that you can print in this state.
	 *
	 * It gets the number of variables that you can print in this state.
	 *
	 * \return The number of variables that you can print in this state.
	 */
	virtual unsigned int GetNumberOfPrintableValues();

	/*!
	 * \brief It gets a value to be printed from this state.
	 *
	 * It gets a value to be printed from this state.
	 *
	 * \param index The synapse's index inside the learning rule.
	 * \param position Position inside each connection.
	 *
	 * \return The value at position-th position in this state.
	 */
	virtual double GetPrintableValuesAt(unsigned int index, unsigned int position);

	/*!
	 * \brief set new time to spikes.
	 *
	 * It set new time to spikes.
	 *
	 * \param index The synapse's index inside the learning rule.
	 * \param NewTime new time.
	 * \param pre_post In some learning rules (i.e. STDPLS) this variable indicate wether the update affects the pre- or post- variables.
	 */
	virtual void SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post);

	/*!
	 * \brief It implements the behaviour when it transmits a spike.
	 *
	 * It implements the behaviour when it transmits a spike. It must be implemented
	 * by any inherited class.
	 *
	 * \param index The synapse's index inside the learning rule.
	 */
	virtual void ApplyPresynapticSpike(unsigned int index);

	/*!
	 * \brief It implements the behaviour when the target cell fires a spike.
	 *
	 * It implements the behaviour when it the target cell fires a spike. It must be implemented
	 * by any inherited class.
	 *
	 * \param index The synapse's index inside the learning rule.
	 */
	virtual void ApplyPostsynapticSpike(unsigned int index);
};

#endif /* ASABUKISYNAPSESTATE_H_ */
