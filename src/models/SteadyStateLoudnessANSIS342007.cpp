/*
 * Copyright (C) 2014 Dominic Ward <contactdominicward@gmail.com>
 *
 * This file is part of Loudness
 *
 * Loudness is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Loudness is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Loudness.  If not, see <http://www.gnu.org/licenses/>. 
 */

#include "../Modules/WeightSpectrum.h"
#include "../Modules/RoexBankANSIS342007.h"
#include "../Modules/SpecificLoudnessGM.h"
#include "../Modules/InstantaneousLoudnessGM.h"
#include "SteadyStateLoudnessANSIS342007.h"

namespace loudness{

    SteadyStateLoudnessANSIS342007::SteadyStateLoudnessANSIS342007() :
        Model("SteadyStateLoudnessANSIS342007", false)
    {
        //Default parameters
        setDiotic(true);
        setDiffuseField(false);
        setFilterSpacing(0.1);
    }

    SteadyStateLoudnessANSIS342007::~SteadyStateLoudnessANSIS342007()
    {
    }

    void SteadyStateLoudnessANSIS342007::setDiotic(bool diotic)
    {
        diotic_ = diotic;
    }

    void SteadyStateLoudnessANSIS342007::setDiffuseField(bool diffuseField)
    {
        diffuseField_ = diffuseField;
    }

    void SteadyStateLoudnessANSIS342007::setFilterSpacing(Real filterSpacing)
    {
        filterSpacing_ = filterSpacing;
    }

    bool SteadyStateLoudnessANSIS342007::initializeInternal(const SignalBank &input)
    {

        /*
         * Weighting filter
         */
        string middleEar = "ANSI";
        string outerEar = "ANSI_FREEFIELD";
        if(diffuseField_)
            outerEar = "ANSI_DIFFUSEFIELD";

        modules_.push_back(unique_ptr<Module>
                (new WeightSpectrum(middleEar, outerEar))); 
        outputNames_.push_back("WeightedPowerSpectrum");

        /*
         * Roex filters
         */
        modules_.push_back(unique_ptr<Module>
                (new RoexBankANSIS342007(1.8, 38.9, filterSpacing_)));
        outputNames_.push_back("ExcitationPattern");
        
        /*
         * Specific loudness using high level modification
         */
        modules_.push_back(unique_ptr<Module>
                (new SpecificLoudnessGM(true)));
        outputNames_.push_back("SpecificLoudnessPattern");

        /*
        * Loudness integration 
        */   
        modules_.push_back(unique_ptr<Module>
                (new InstantaneousLoudnessGM(1.0, true)));
        outputNames_.push_back("InstantaneousLoudness");

        //configure targets
        setUpLinearTargetModuleChain();

        return 1;
    }
}