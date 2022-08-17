using System;
using System.Collections.Generic;
using System.Drawing;
using NeatNetwork.Libraries;

namespace NeatNetwork.NetworkFiles
{
    public class NeuronConnectionsInfo
    {
        public int Length => Weights.Count;
        internal List<Point> ConnectedNeuronsPos { get; private set; }
        internal List<double> Weights;

        internal NeuronConnectionsInfo()
        {
            Weights = new List<double>();
            ConnectedNeuronsPos = new List<Point>();
        }

        internal NeuronConnectionsInfo(List<Point> connectedNeuronsPos, List<double> weights)
        {
            this.ConnectedNeuronsPos = connectedNeuronsPos;
            this.Weights = weights;
        }

        /// <summary>
        /// Layer 0 is input layer
        /// </summary>
        internal void AddNewConnection(int layerIndex, int neuronIndex, double minValue, double maxValue, double valueClosestTo0)
        {
            AddNewConnection(layerIndex, neuronIndex, ValueGeneration.GenerateWeight(minValue, maxValue, valueClosestTo0));
        }

        /// <summary>
        /// Layer 0 is input layer
        /// </summary>
        internal void AddNewConnection(int layerIndex, int neuronIndex, double weight)   
        {
            ConnectedNeuronsPos.Add(new Point(layerIndex, neuronIndex));
            Weights.Add(weight);
        }

        internal void AdjustToNewLayerBeingAdded(int layerInsertionIndex, bool isinsertedInPreviousLayer, int insertedLayerLength, double minWeight, double maxWeight, double weightClosestTo0)
        {
            for (int i = 0; i < ConnectedNeuronsPos.Count; i++)
                ConnectedNeuronsPos[i] = new Point(ConnectedNeuronsPos[i].X + 1 * Convert.ToInt32(ConnectedNeuronsPos[i].X > layerInsertionIndex + 1), ConnectedNeuronsPos[i].Y);

            if (!isinsertedInPreviousLayer)
                return;

            for (int i = 0; i < insertedLayerLength; i++)
            {
                //            +1 makes input layer count
                AddNewConnection(layerInsertionIndex + 1, i, minWeight, maxWeight, weightClosestTo0);
            }
        }

        internal void SubtractGrads(List<double> weightGradients, double learningRate)
        {
            for (int i = 0; i < Weights.Count; i++)
                Weights[i] -= weightGradients[i] * learningRate;
        }

        internal new string ToString()
        {
            string str = "";
            for (int i = 0; i < Weights.Count; i++)
            {
                str += $"Pos: {ConnectedNeuronsPos[i].X}*{ConnectedNeuronsPos[i].Y}* Weight: {Weights[i]}|";
            }

            return str;
        }

        public NeuronConnectionsInfo(string str)
        {
            ConnectedNeuronsPos = new List<Point>();
            Weights = new List<double>();

            str = str.Replace("Pos: ", "").Replace(" Weight: ", "");
            string[] strs = str.Split(new char[] { '|' }, StringSplitOptions.RemoveEmptyEntries);
            foreach (var connectionStr in strs)
            {
                string[] currentConnectionFields = connectionStr.Split(new char[] { '*' });
                ConnectedNeuronsPos.Add(new Point(Convert.ToInt32(currentConnectionFields[0]), Convert.ToInt32(currentConnectionFields[1])));
                Weights.Add(Convert.ToDouble(currentConnectionFields[2]));
            }
        }
    }
}
