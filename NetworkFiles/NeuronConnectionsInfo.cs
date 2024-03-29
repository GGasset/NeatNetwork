﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Threading;
using System.Threading.Tasks;
using NeatNetwork.Libraries;

namespace NeatNetwork.NetworkFiles
{
    public class NeuronConnectionsInfo
    {
        public int Length => Weights.Count;
        internal List<Point> ConnectedNeuronsPos;
        internal List<double> Weights;

        internal NeuronConnectionsInfo()
        {
            Weights = new List<double>();
            ConnectedNeuronsPos = new List<Point>();
        }

        private class PositionGenerator
        {
            private int connectedX, startingY, outputLength;

            internal PositionGenerator(int connectedX, int startingY, int outputLength)
            {
                this.connectedX = connectedX;
                this.startingY = startingY;
                this.outputLength = outputLength;
            }

            internal Task<List<Point>> RunAsync() => Task.Run(() => ValueGeneration.GetConnectionsConnectedPosition(connectedX, startingY, outputLength));
        }

        internal NeuronConnectionsInfo(int layerIndex, int previousLayerLength, double minWeight, double maxWeight, double valueClosestTo0)
        {
            Weights = new List<double>();
            ConnectedNeuronsPos = new List<Point>();

            int connectionsPerTask = 2000;

            if (previousLayerLength > connectionsPerTask)
            {
                // specify task job
                int taskCount = previousLayerLength / connectionsPerTask;
                int leftConnectionCount = previousLayerLength % connectionsPerTask;

                // Initialize tasks
                List<Task<List<double>>> weigthsTasks = new List<Task<List<double>>>();
                List<Task<List<Point>>> positionsTasks = new List<Task<List<Point>>>();
                PositionGenerator[] taskGenerator = new PositionGenerator[taskCount];

                for (int i = 0; i < taskCount; i++)
                {
                    weigthsTasks.Add(Task.Run(() => ValueGeneration.GenerateWeights(connectionsPerTask, minWeight, maxWeight, valueClosestTo0)));

                    taskGenerator[i] = new PositionGenerator(layerIndex - 1, i * connectionsPerTask, connectionsPerTask);
                    positionsTasks.Add(taskGenerator[i].RunAsync());
                }

                weigthsTasks.Add(Task.Run(() => ValueGeneration.GenerateWeights(leftConnectionCount, minWeight, maxWeight, valueClosestTo0)));
                positionsTasks.Add(Task.Run(() => ValueGeneration.GetConnectionsConnectedPosition(layerIndex - 1, taskCount * connectionsPerTask, leftConnectionCount)));

                // wait for tasks to finish execution
                foreach (var weightsTask in weigthsTasks)
                {
                    weightsTask.Wait();
                }
                foreach (var positionsTask in positionsTasks)
                {
                    positionsTask.Wait();
                }

                // Initialize connections
                for (int i = 0; i < weigthsTasks.Count; i++)
                {
                    AddConnections(positionsTasks[i].Result, weigthsTasks[i].Result);
                }
                return;
            }

            for (int i = 0; i < previousLayerLength; i++)
            {
                Point connectionPos = new Point(layerIndex - 1, i);
                AddNewConnection(connectionPos, ValueGeneration.GenerateWeight(minWeight, maxWeight, valueClosestTo0));
            }
        }

        internal NeuronConnectionsInfo(List<Point> connectedNeuronsPos, List<double> weights)
        {
            this.ConnectedNeuronsPos = connectedNeuronsPos;
            this.Weights = weights;
        }

        #region Evolution Learning

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

        internal void AddNewConnection(Point connectionPos, double weight)
        {
            ConnectedNeuronsPos.Add(connectionPos);
            Weights.Add(weight);
        }

        internal void AddConnections(List<Point> positions, List<double> weigths)
        {
            ConnectedNeuronsPos.AddRange(positions);
            Weights.AddRange(weigths);
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

        internal void Evolve(double maxVariation, double mutationChance)
        {
            for (int i = 0; i < Weights.Count; i++)
            {
                Weights[i] += ValueGeneration.EvolveValue(maxVariation, mutationChance);
            }
        }

        #endregion

        #region Gradient Learning

        internal void SubtractGrads(NeuronConnectionsInfo gradients, double learningRate) => SubtractGrads(gradients.Weights, learningRate);

        internal void SubtractGrads(List<double> weightGradients, double learningRate)
        {
            for (int i = 0; i < Weights.Count; i++)
                Weights[i] -= weightGradients[i] * learningRate;
        }

        #endregion


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

        public NeuronConnectionsInfo Clone() => new NeuronConnectionsInfo(ConnectedNeuronsPos, Weights);
    }
}
