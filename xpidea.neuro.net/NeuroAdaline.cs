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


using System.IO;
using xpidea.neuro.net.patterns;

namespace xpidea.neuro.net.adaline
{
    /// <summary>
    ///     Class implementing an Adaline node in the Adaline network.
    /// </summary>
    public class AdalineNode : FeedForwardNode
    {
        /// <summary>
        ///     Stores the learning rate of the node.
        /// </summary>
        private double nodeLearningRate;

        /// <summary>
        ///     Initializes a new instance of the Adaline node.
        /// </summary>
        public AdalineNode()
        {
        }

        /// <summary>
        ///     Initializes a new instance of the Adaline node and sets the
        ///     <see cref="xpidea.neuro.net.adaline.AdalineNode.LearningRate" />.
        /// </summary>
        /// <param name="learningRate">Usually in range  from 0..1 </param>
        public AdalineNode(double learningRate)
        {
            nodeLearningRate = learningRate;
        }

        /// <summary>
        ///     Gets and sets the LearnigRate value of the Adaline node.
        /// </summary>
        /// <remarks>The LearningRate value defines how fast node will "learn".</remarks>
        public double LearningRate
        {
            get { return GetNodeLearningRate(); }
            set { SetNodeLearningRate(value); }
        }

        /// <summary>
        ///     Gets <see cref="xpidea.neuro.net.adaline.AdalineNode.LearningRate" /> of the node.
        /// </summary>
        /// <returns>Learning rate value.</returns>
        protected virtual double GetNodeLearningRate()
        {
            return nodeLearningRate;
        }

        /// <summary>
        ///     Sets <see cref="xpidea.neuro.net.adaline.AdalineNode.LearningRate" /> of the node.
        /// </summary>
        /// <param name="learningRate">Learnig rate value.</param>
        protected virtual void SetNodeLearningRate(double learningRate)
        {
            nodeLearningRate = learningRate;
        }

        /// <summary>
        ///     Overridden. Translates node output into network specific output. In case of Adaline network implements simple
        ///     threshold function: Returns -1 if value less than 0, otherwise returns 1.
        /// </summary>
        /// <param name="value">Node value.</param>
        /// <returns>Result value.</returns>
        protected override double Transfer(double value)
        {
            if (value < 0)
                return -1;
            return 1;
        }

        /// <summary>
        ///     Overridden. Implements the Delta Rule to modify the link values.
        /// </summary>
        /// <remarks>This method gets executed only if the node produces incorrect output.</remarks>
        public override void Learn()
        {
            Error = Value * -2;
            foreach (var link in InLinks)
            {
                var delta = LearningRate * link.InNode.Value * Error;
                link.UpdateWeight(delta);
            }
        }
    }

    /// <summary>
    ///     Represents the link in the <see cref="xpidea.neuro.net.adaline.AdalineNetwork" /> network.
    /// </summary>
    public class AdalineLink : NeuroLink
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="xpidea.neuro.net.adaline.AdalineLink" /> and sets
        ///     <see cref="xpidea.neuro.net.NeuroLink.Weight" /> to a random value of range from -1 to 1.
        /// </summary>
        public AdalineLink()
        {
            Weight = Random(-1, 1);
        }
    }

    /// <summary>
    ///     Represents an Adaline network.
    /// </summary>
    /// <remarks>
    ///     An adaptive linear element or Adaline, proposed by Widrow (1959, 1960), is a simple perceptron-like system that
    ///     accomplishes classification by modifying weights in such a way as to diminish the mean squared error (MSE) at every
    ///     iteration. The architecture of the Adaline is the simplest of all neural networks. It is a simple processing
    ///     element capable of sorting a set of input patterns into two categories. It has an ability to learn through a
    ///     supervised learning process.
    ///     Although the Adaline works quite well for many applications, it is restrictd to a linear problem space. The input
    ///     patterns in the Adaline's training set must be linearly separable; otherwise, the Adaline will never categorize all
    ///     of the training patterns correctly even when it reaches the low point of the error surface paraboloid. However, the
    ///     Adaline is guaranteed to reach its minimum error state since there are no obstacles along the error surface (like
    ///     local minima) to interfere with the training process.Training occurs by repeatedly presenting sets of data composed
    ///     of input patterns and their desired outputs. Learning occurs as the Adaline minimized the number of errors it makes
    ///     when sorting the patterns into their correct categories. Once trained, the Adaline can categorize new inputs
    ///     according to the experience it gained.
    ///     <img src="Adaline.jpg"></img>
    /// </remarks>
    public class AdalineNetwork : NeuralNetwork
    {
        /// <summary>
        ///     Stores a learning rate value.
        /// </summary>
        protected double learningRate;

        /// <summary>
        ///     Creates an instance of Adaline network.
        /// </summary>
        /// <param name="aNodesCount">Number of input nodes of the network.</param>
        /// <param name="learningRate">Learning rate.</param>
        public AdalineNetwork(int aNodesCount, double learningRate)
        {
            nodesCount = aNodesCount + 2;
            linksCount = aNodesCount + 1;
            this.learningRate = learningRate;
            CreateNetwork();
        }

        /// <summary>
        ///     Creates unitialized instance of Adaline network.
        /// </summary>
        public AdalineNetwork()
        {
        }

        /// <summary>
        ///     Creates the network from a file.
        /// </summary>
        public AdalineNetwork(string fileName) : base(fileName)
        {
        }

        /// <summary>
        ///     Network's learning rate property.
        /// </summary>
        public double LearningRate
        {
            get { return learningRate; }
        }

        /// <summary>
        ///     An output, Adaline node, of the Adaline network.
        /// </summary>
        public AdalineNode AdalineNode
        {
            get { return GetAdalineNode(); }
        }

        private AdalineNode GetAdalineNode()
        {
            return (AdalineNode)(OutputNode(OutputNodesCount - 1));
        }

        /// <summary>
        ///     Overridden.Constructs network topology.
        /// </summary>
        /// <remarks>Creates nodes, links and connects nodes using created links.</remarks>
        protected override void CreateNetwork()
        {
            nodes = new NeuroNode[NodesCount];
            links = new NeuroLink[LinksCount];
            for (var i = 0; i < InputNodesCount; i++)
                nodes[i] = new InputNode();
            nodes[NodesCount - 2] = new BiasNode(1);
            nodes[NodesCount - 1] = new AdalineNode(LearningRate);
            for (var i = 0; i < LinksCount; i++)
                links[i] = new AdalineLink();
            for (var i = 0; i < LinksCount; i++)
                nodes[i].LinkTo(nodes[NodesCount - 1], links[i]);
        }

        /// <summary>
        ///     Overridden.Returns type of the network.
        /// </summary>
        /// <returns>Returns <see cref="xpidea.neuro.net.NeuralNetworkType.nnAdaline" /> for Adaline networks.</returns>
        /// <remarks>Used for persistence purposes.</remarks>
        protected override NeuralNetworkType GetNetworkType()
        {
            return NeuralNetworkType.nnAdaline;
        }

        /// <summary>
        ///     Overridden.Returns number of Input nodes in the network.
        /// </summary>
        /// <returns>Number of inputs.</returns>
        protected override int GetInputNodesCount()
        {
            return NodesCount - 2;
        }

        /// <summary>
        ///     Overridden.Returns number of output nodes in the network.
        /// </summary>
        /// <returns>Number of output nodes. In case of Adaline network it always 1.</returns>
        protected override int GetOutPutNodesCount()
        {
            return 1;
        }

        /// <summary>
        ///     Overridden.Retrieves an input node by its index.
        /// </summary>
        /// <param name="index">Node index</param>
        /// <returns>Input node.</returns>
        protected override NeuroNode GetInputNode(int index)
        {
            if ((index >= InputNodesCount) || (index < 0))
                throw new ENeuroException("InputNode index out of bounds.");
            return nodes[index];
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
            return nodes[NodesCount - 1];
        }

        /// <summary>
        ///     Sets input values of the network from the pattern.
        /// </summary>
        /// <param name="pattern">Training pattern.</param>
        public virtual void SetValuesFromPattern(Pattern pattern)
        {
            for (var i = 0; i < pattern.InputsCount; i++)
                nodes[i].Value = pattern.Input[i];
        }

        /// <summary>
        ///     Overridden.Performs network training. Here you write the code to train your network.
        /// </summary>
        /// <param name="patterns">Set of the patterns that will be exposed to a network during the training.</param>
        /// <remarks>
        ///     This method implementation is for reference only -
        ///     You may want to implement your own method by overriding this one.
        ///     This implementation will
        ///     complete network training only after the network will produce
        ///     correct output for all input patterns.
        ///     Be advised that in this example network training will never complete if input patterns
        ///     have non-linear character.
        /// </remarks>
        public override void Train(PatternsCollection patterns)
        {
            int Good, i;
            if (patterns != null)
            {
                Good = 0;
                while (Good < patterns.Count)
                {
                    Good = 0;
                    for (i = 0; i < patterns.Count; i++)
                    {
                        SetValuesFromPattern(patterns[i]);
                        AdalineNode.Run();
                        if ((patterns[i]).Output[0] != AdalineNode.Value)
                        {
                            AdalineNode.Learn();
                            break;
                        }
                        Good++;
                    }
                }
            }
        }

        /// <summary>
        ///     Overridden.Loads network data from a binary stream.
        /// </summary>
        /// <param name="binaryReader">Binary stream reader.</param>
        public override void Load(BinaryReader binaryReader)
        {
            base.Load(binaryReader);
            learningRate = binaryReader.ReadDouble();
        }

        /// <summary>
        ///     Overridden.Saves the network to a binary stream.
        /// </summary>
        /// <param name="binaryWriter">Binary stream writer.</param>
        public override void Save(BinaryWriter binaryWriter)
        {
            base.Save(binaryWriter);
            binaryWriter.Write(learningRate);
        }
    }
}