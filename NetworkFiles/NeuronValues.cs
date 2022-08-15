using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork.NetworkFiles
{
    public class NeuronValues
    {
        internal readonly NeuronHolder.NeuronType NeuronType;
        internal double Activation;
        internal double LinearFunction;
        internal double HiddenState;
        internal double CellState;

        internal NeuronValues(NeuronHolder.NeuronType neuronType)
        {
            NeuronType = neuronType;
            Activation = 0;
            LinearFunction = 0.0;
            HiddenState = 0.0;
            CellState = 0.0;
        }
    }
}
