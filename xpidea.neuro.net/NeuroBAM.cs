/*

  Neuro.NET - Library of neural networks for .NET
  Copyright (C) 2001-2015  Alex Cherkasov. All rights reserved.
                           email: info@xpidea.com
                           web: http://xpidea.com/

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2
  of the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
  02110-1301, USA.

{   CREDITS:                                                    }
{   This work is based on publications of:                      }
{          -Christopher M. Bishop                               }
{          -Jose C. Principe                                    }
{          -Samuel J. Rogers                                    }
{          -Laurene V. Fausett                                  }
{          -Simon S. Haykin                                     }
  
*/

using System;
using System.IO;
using xpidea.neuro.net.adaline;
using xpidea.neuro.net.patterns;

namespace xpidea.neuro.net.bam
{
    /// <summary>
    ///     A link for the Bidirectional Associative Memory (BAM) network.
    /// </summary>
    public class BidirectionalAssociativeMemoryLink : AdalineLink
    {
        /// <summary>
        ///     Constructs the link and initializes its weight to zero.
        /// </summary>
        public BidirectionalAssociativeMemoryLink()
        {
            Weight = 0;
        }
    }

    /// <summary>
    ///     Implements output node in the BAM network.
    /// </summary>
    public class BidirectionalAssociativeMemoryOutputNode : AdalineNode
    {
        /// <summary>
        ///     Stores previous value of the node.
        /// </summary>
        protected double nodesLastValue;

        /// <summary>
        ///     Property contains previous (the value of the node before method
        ///     <see cref="xpidea.neuro.net.bam.BidirectionalAssociativeMemoryOutputNode.Run" /> was executed) value of the node.
        /// </summary>
        public double NodeLastValue
        {
            get { return GetNodeLastValue(); }
            set { SetNodeLastValue(value); }
        }

        /// <summary>
        ///     Retrieves previous value of the node.
        /// </summary>
        /// <returns>Node previous value.</returns>
        protected virtual double GetNodeLastValue()
        {
            return nodesLastValue;
        }

        /// <summary>
        ///     Sets node previous value.
        /// </summary>
        /// <param name="aLastValue">Previous value of the node.</param>
        protected virtual void SetNodeLastValue(double aLastValue)
        {
            nodesLastValue = aLastValue;
        }

        /// <summary>
        ///     Overridden.Sets node value.
        /// </summary>
        /// <param name="value">New node value.</param>
        /// <remarks>As well sets node error equal to node value.</remarks>
        protected override void SetNodeValue(double value)
        {
            nodeError = value;
            nodeValue = value;
        }

        /// <summary>
        ///     Overridden.Runs the node.
        /// </summary>
        /// <remarks>
        ///     Stores current value of the node as
        ///     <see cref="xpidea.neuro.net.bam.BidirectionalAssociativeMemoryOutputNode.NodeLastValue" /> and runs the node to
        ///     calculate the new value.
        /// </remarks>
        public override void Run()
        {
            NodeLastValue = Value;
            base.Run();
        }

        /// <summary>
        ///     Overridden.Teaches the node.
        /// </summary>
        public override void Learn()
        {
            foreach (var link in InLinks)
                link.UpdateWeight(link.InNode.Value*link.OutNode.Value);
        }

        /// <summary>
        ///     Overridden.Loads node data from a binary stream.
        /// </summary>
        /// <param name="binaryReader">Binary stream reader.</param>
        public override void Load(BinaryReader binaryReader)
        {
            base.Load(binaryReader);
            nodesLastValue = binaryReader.ReadDouble();
        }

        /// <summary>
        ///     Overridden.Stores node data into the binary stream.
        /// </summary>
        /// <param name="binaryWriter">Binary stream writer.</param>
        public override void Save(BinaryWriter binaryWriter)
        {
            base.Save(binaryWriter);
            binaryWriter.Write(nodesLastValue);
        }
    }

    /// <summary>
    ///     Implements an input node in BAM network.
    /// </summary>
    public class BidirectionalAssociativeMemoryInputNode : BidirectionalAssociativeMemoryOutputNode
    {
        /// <summary>
        ///     Overridden.Runs the node.
        /// </summary>
        public override void Run()
        {
            NodeLastValue = Value;
            double total = 0;
            foreach (var link in OutLinks)
                total += link.WeightedOutValue();
            nodeValue = Transfer(total);
        }

        /// <summary>
        ///     Overridden.Does nothing, since the learning process of the BAM network is different.
        /// </summary>
        public override void Learn()
        {
        }
    }

    /// <summary>
    ///     Implements the Bidirectional Associative Memory  (BAM) network.
    /// </summary>
    /// <remarks>
    ///     <img src="BAM.jpg"></img>
    ///     The  Bidirectional Associative Memory (BAM) network that consists of two layers. An input layer and an output
    ///     layer. The main difference between backpropagation architecture is that BAM does not stop learning when input
    ///     values reach the output layer. The learning phase stops when the network becomes stable ; no change between input
    ///     and output values during two consecutive cycles. The pattern sets for training and running and ouput results can
    ///     only have two values : 1.1 or -1.1. The BAM is a relatively simple neural network architecture with a content
    ///     addressable memory. BAM is usefull for pattern recognition or with noisy and corrupted patterns.  Bam can also
    ///     "forget" if there are two many patterns in it. BAM becomes saturated when the number of patterns stored is greater
    ///     than the minimum of the input layer count and the ouput layer node count. BAM System is created to solve this
    ///     problem.
    /// </remarks>
    public class BidirectionalAssociativeMemoryNetwork : AdalineNetwork
    {
        /// <summary>
        ///     Stores nodes count in input layer.
        /// </summary>
        protected int inputLayerNodesCount;

        /// <summary>
        ///     Stores nodes count in output layer.
        /// </summary>
        protected int outputLayerNodesCount;

        /// <summary>
        ///     Creates BAM network.
        /// </summary>
        /// <param name="aInputNodesCount">Number of nodes in the input layer.</param>
        /// <param name="aOutputNodesCount">Number of nodes in the output layer.</param>
        public BidirectionalAssociativeMemoryNetwork(int aInputNodesCount, int aOutputNodesCount)
        {
            inputLayerNodesCount = aInputNodesCount;
            outputLayerNodesCount = aOutputNodesCount;
            nodesCount = InputNodesCount + OutputNodesCount;
            linksCount = InputNodesCount*OutputNodesCount;
            CreateNetwork();
        }

        /// <summary>
        ///     Creates the network from a file.
        /// </summary>
        public BidirectionalAssociativeMemoryNetwork(string fileName) : base(fileName)
        {
        }

        /// <summary>
        ///     Returns value of the output node specified by index.
        /// </summary>
        /// <param name="index">Node index.</param>
        /// <returns>Node value.</returns>
        public virtual double value(int index)
        {
            return nodes[InputNodesCount + index].Value;
        }

        /// <summary>
        ///     Overridden.Returns node error.
        /// </summary>
        /// <returns>Node error.</returns>
        protected override double GetNodeError()
        {
            double e1 = 0;
            double e2 = 0;
            for (var i = InputNodesCount; i < InputNodesCount + OutputNodesCount; i++)
                for (var j = 0; j < nodes[i].InLinks.Count; j++)
                {
                    var node = nodes[i];
                    var link = node.InLinks[j];
                    e1 = e1 + link.WeightedInValue()*link.OutNode.Value;
                    e2 = e2 + link.WeightedInError()*link.OutNode.Value;
                }
            if (e1 == e2)
                return Math.Abs(-InputNodesCount*OutputNodesCount + e1);
            return double.PositiveInfinity;
        }

        /// <summary>
        ///     Sets value of the node.
        /// </summary>
        /// <param name="index">Node index.</param>
        /// <param name="value">New node value.</param>
        public virtual void SetValues(int index, double value)
        {
            nodes[index].Value = value;
        }

        /// <summary>
        ///     Overridden.Constucts the network topology.
        /// </summary>
        protected override void CreateNetwork()
        {
            nodes = new NeuroNode[NodesCount];
            links = new NeuroLink[LinksCount];
            for (var i = 0; i < InputNodesCount; i++)
                nodes[i] = new BidirectionalAssociativeMemoryInputNode();
            for (var i = InputNodesCount; i < InputNodesCount + OutputNodesCount; i++)
                nodes[i] = new BidirectionalAssociativeMemoryOutputNode();
            for (var i = 0; i < linksCount; i++)
                links[i] = new BidirectionalAssociativeMemoryLink();
            var k = 0;

            for (var i = 0; i < InputNodesCount; i++)
                for (var j = InputNodesCount; j < InputNodesCount + OutputNodesCount; j++)
                {
                    nodes[i].LinkTo(nodes[j], links[k]);
                    k++;
                }
        }

        /// <summary>
        ///     Overridden.Loads input values into the input layer.
        /// </summary>
        protected override void LoadInputs()
        {
            for (var i = 0; i < InLinks.Count; i++)
                SetValues(i, InLinks[i].InNode.Value);
        }

        /// <summary>
        ///     Overridden.Returns <see cref="xpidea.neuro.net.NeuralNetworkType.nntBAM" /> for the BAM network.
        /// </summary>
        /// <returns></returns>
        protected override NeuralNetworkType GetNetworkType()
        {
            return NeuralNetworkType.nntBAM;
        }

        /// <summary>
        ///     Overridden.Returns nodes count in the input layer.
        /// </summary>
        /// <returns>Nodes count.</returns>
        protected override int GetInputNodesCount()
        {
            return inputLayerNodesCount;
        }

        /// <summary>
        ///     Overridden.Returns nodes count in output layer.
        /// </summary>
        /// <returns>Nodes count.</returns>
        protected override int GetOutPutNodesCount()
        {
            return outputLayerNodesCount;
        }

        /// <summary>
        ///     Overridden.Returns output node by its index.
        /// </summary>
        /// <param name="index">Output node index.</param>
        /// <returns>Node.</returns>
        protected override NeuroNode GetOutputNode(int index)
        {
            if ((index >= OutputNodesCount) || (index < 0))
                throw new ENeuroException("OutputNode index out of bounds.");
            return nodes[index + InputNodesCount];
        }

        /// <summary>
        ///     Overridden.Loads the values into the input layer from the pattern.
        /// </summary>
        /// <param name="pattern">Pattern.</param>
        public override void SetValuesFromPattern(Pattern pattern)
        {
            for (var i = 0; i < pattern.InputsCount; i++)
                nodes[i].Value = pattern.Input[i];
            for (var i = 0; i < pattern.OutputsCount; i++)
                nodes[i + InputNodesCount].Value = pattern.Output[i];
        }

        /// <summary>
        ///     Overridden.Runs the network.
        /// </summary>
        public override void Run()
        {
            LoadInputs();
            var IsStable = false;
            var iteration = 0;
            while (!IsStable)
            {
                IsStable = true;
                iteration++;
                for (var i = InputNodesCount + OutputNodesCount - 1; i >= 0; i--)
                    nodes[i].Run();
                if (iteration > 1)
                {
                    for (var j = 0; j < InputNodesCount + OutputNodesCount; j++)
                    {
                        var node = (BidirectionalAssociativeMemoryOutputNode) nodes[j];
                        if (!IsStable) break;
                        if (node.Value != node.NodeLastValue)
                            IsStable = false;
                    }
                }
                else
                    IsStable = false;
            }
        }

        /// <summary>
        ///     Overridden.Teaches the network.
        /// </summary>
        public override void Learn()
        {
            for (var i = InputNodesCount; i < InputNodesCount + OutputNodesCount; i++)
                nodes[i].Learn();
        }

        /// <summary>
        ///     Overridden.Trains the network to recognize specific patterns. Employs
        ///     <see cref="xpidea.neuro.net.bam.BidirectionalAssociativeMemoryOutputNode.Run" /> and
        ///     <see cref="xpidea.neuro.net.bam.BidirectionalAssociativeMemoryOutputNode.Learn" />
        ///     to teach the network.
        /// </summary>
        /// <param name="patterns">Training patterns.</param>
        public override void Train(PatternsCollection patterns)
        {
            if (patterns != null)
                for (var i = 0; i < patterns.Count; i++)
                {
                    SetValuesFromPattern(patterns[i]);
                    Learn();
                }
        }

        /// <summary>
        ///     Tells the network to "forget" last learn operation.
        /// </summary>
        public void UnLearn()
        {
            for (var i = InputNodesCount; i < InputNodesCount + OutputNodesCount; i++)
                nodes[i].Value = -nodes[i].Value;
            Learn();
        }

        /// <summary>
        ///     Overridden.Loads network data from a binary stream.
        /// </summary>
        /// <param name="binaryReader">Binary stream reader.</param>
        public override void Load(BinaryReader binaryReader)
        {
            inputLayerNodesCount = binaryReader.ReadInt32();
            outputLayerNodesCount = binaryReader.ReadInt32();
            base.Load(binaryReader);
        }

        /// <summary>
        ///     Overridden.Stores network data in the binary stream.
        /// </summary>
        /// <param name="binaryWriter">Binary stream writer.</param>
        public override void Save(BinaryWriter binaryWriter)
        {
            binaryWriter.Write(inputLayerNodesCount);
            binaryWriter.Write(outputLayerNodesCount);
            base.Save(binaryWriter);
        }
    }

    /// <summary>
    ///     Implements the system of BAM networks.
    /// </summary>
    public class BidirectionalAssociativeMemorySystem : BidirectionalAssociativeMemoryNetwork
    {
        /// <summary>
        ///     The network having the minimum error.
        /// </summary>
        protected BidirectionalAssociativeMemoryNetwork best;

        /// <summary>
        ///     Best error.
        /// </summary>
        protected double bestError;

        /// <summary>
        ///     Stores the pattern.
        /// </summary>
        protected Pattern data;

        /// <summary>
        ///     Array of BAM networks.
        /// </summary>
        protected BidirectionalAssociativeMemoryNetwork[] networks;

        /// <summary>
        ///     Stores networks count in the system.
        /// </summary>
        protected int networksCount;

        /// <summary>
        ///     Orthogonal network energy.
        /// </summary>
        protected double orthogonalBAMEnergy;

        /// <summary>
        ///     Creates the BAM system.
        /// </summary>
        /// <param name="aInputNodesCount">Number of nodes in input layer.</param>
        /// <param name="aOutputNodesCount">Number of nodes in output layer.</param>
        public BidirectionalAssociativeMemorySystem(int aInputNodesCount, int aOutputNodesCount)
            : base(aInputNodesCount, aOutputNodesCount)
        {
            Create(aInputNodesCount, aOutputNodesCount);
        }

        /// <summary>
        ///     Overridden.Node value by index from the best network.
        /// </summary>
        /// <param name="index">Node index.</param>
        /// <returns>Node value.</returns>
        public override double value(int index)
        {
            return best.value(index);
        }

        /// <summary>
        ///     Overridden.Returns the best error.
        /// </summary>
        /// <returns>Error value.</returns>
        protected override double GetNodeError()
        {
            return bestError;
        }

        /// <summary>
        ///     Overridden.Stores node values in the <see cref="xpidea.neuro.net.bam.BidirectionalAssociativeMemorySystem.data" />
        ///     field.
        /// </summary>
        /// <param name="index">Node index.</param>
        /// <param name="value">Node value.</param>
        public override void SetValues(int index, double value)
        {
            data.Input[index] = value;
        }

        /// <summary>
        ///     Overridden.Initializes network data after construction.
        /// </summary>
        protected override void CreateNetwork()
        {
            nodes = new NeuroNode[0];
            links = new NeuroLink[0];
            networksCount = 0;
            orthogonalBAMEnergy = -inputLayerNodesCount*outputLayerNodesCount;
            bestError = double.PositiveInfinity;
        }

        /// <summary>
        ///     Overridden.Loads input data to the system from the the nodes attached to input.
        /// </summary>
        protected override void LoadInputs()
        {
            for (var i = 0; i < InLinks.Count; i++)
                data.Input[i] = InLinks[i].InNode.Value;
        }

        /// <summary>
        ///     Overridden.Returns <see cref="xpidea.neuro.net.NeuralNetworkType.nntBAMSystem" /> for BAM system.
        /// </summary>
        /// <returns>Type of neural network.</returns>
        protected override NeuralNetworkType GetNetworkType()
        {
            return NeuralNetworkType.nntBAMSystem;
        }

        private void Create(int aInputNodesCount, int aOutputNodesCount)
        {
            inputLayerNodesCount = aInputNodesCount;
            outputLayerNodesCount = aOutputNodesCount;
            CreateNetwork();
            data = new Pattern(aInputNodesCount, aOutputNodesCount);
        }

        /// <summary>
        ///     Overridden.Runs the system.
        /// </summary>
        public override void Run()
        {
            LoadInputs();
            bestError = double.PositiveInfinity;
            for (var i = 0; i < networksCount; i++)
            {
                networks[i].SetValuesFromPattern(data);
                networks[i].Run();
                var err = networks[i].Error;
                if (err <= bestError)
                {
                    bestError = err;
                    best = networks[i];
                }
            }
        }

        /// <summary>
        ///     Overridden.Teaches the system.
        /// </summary>
        public override void Learn()
        {
            var isLearned = false;
            for (var i = 0; i < networksCount; i++)
            {
                if (isLearned) break;
                networks[i].SetValuesFromPattern(data);
                networks[i].Learn();
                if (networks[i].Error != 0)
                    networks[i].UnLearn();
                else
                    isLearned = true;
            }
            if (!isLearned)
            {
                var oldNetworks = networks;
                networks = new BidirectionalAssociativeMemoryNetwork[networksCount + 1];
                if (oldNetworks != null)
                    oldNetworks.CopyTo(networks, 0);
                networks[networksCount] = new BidirectionalAssociativeMemoryNetwork(inputLayerNodesCount,
                    outputLayerNodesCount);
                networks[networksCount].SetValuesFromPattern(data);
                networks[networksCount].Learn();
                networksCount++;
            }
        }

        /// <summary>
        ///     Overridden.Loads the BAM system data from a binary stream.
        /// </summary>
        /// <param name="binaryReader">Binary stream reader.</param>
        public override void Load(BinaryReader binaryReader)
        {
            base.Load(binaryReader);
            Create(inputLayerNodesCount, outputLayerNodesCount);
            networksCount = binaryReader.ReadInt32();
            var networks = new BidirectionalAssociativeMemoryNetwork[networksCount];
            for (var i = 0; i < networksCount; i++)
            {
                networks[i] = new BidirectionalAssociativeMemoryNetwork(inputLayerNodesCount, outputLayerNodesCount);
                networks[i].Load(binaryReader);
            }
        }

        /// <summary>
        ///     Overridden.Stores BAM system data in a binary stream.
        /// </summary>
        /// <param name="binaryWriter">Binary stream writer.</param>
        public override void Save(BinaryWriter binaryWriter)
        {
            base.Save(binaryWriter);
            binaryWriter.Write(networksCount);
            for (var i = 0; i < networksCount; i++)
                networks[i].Save(binaryWriter);
        }

        /// <summary>
        ///     Overridden.Retrieves an output node by its index.
        /// </summary>
        /// <param name="index">Node index.</param>
        /// <returns>Output node.</returns>
        protected override NeuroNode GetOutputNode(int index)
        {
            if ((index >= OutputNodesCount) || (index < 0))
                throw new ENeuroException("OutputNode index out of bounds.");
                    //In case of Adaline an index always will be 0.
            return best.OutputNode(index);
        }

        /// <summary>
        ///     Overridden.Sets input and output values from the pattern.
        /// </summary>
        /// <param name="pattern"></param>
        public override void SetValuesFromPattern(Pattern pattern)
        {
            for (var i = 0; i < pattern.InputsCount; i++)
                data.Input[i] = pattern.Input[i];
            for (var i = 0; i < pattern.OutputsCount; i++)
                data.Output[i] = pattern.Output[i];
        }
    }
}