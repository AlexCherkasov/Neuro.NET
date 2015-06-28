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
using System.Collections;
using System.IO;
using xpidea.neuro.net.patterns;

namespace xpidea.neuro.net
{
    /// <summary>
    ///     Specifies the type of a neural network, a property of a NeuralNetwork used for persitence purposes.
    /// </summary>
    public enum NeuralNetworkType
    {
        /// <summary>
        ///     Adaline network.
        /// </summary>
        nnAdaline,

        /// <summary>
        ///     Backpropagation network.
        /// </summary>
        nntBackProp,

        /// <summary>
        ///     Self Organizing network
        /// </summary>
        nntSON,

        /// <summary>
        ///     Bidirectional Associative Memory network.
        /// </summary>
        nntBAM,

        /// <summary>
        ///     System of Bidirectional Associative Memory networks.
        /// </summary>
        nntBAMSystem,

        /// <summary>
        ///     Backpropagation network implementing epoch learning.
        /// </summary>
        nntEpochBackProp
    };

    /// <summary>
    ///     Is the abstract base class for all neural network classes.
    /// </summary>
    public abstract class NeuroObject
    {
        private static bool infoShown;
        private static readonly Random random = new Random();

        /// <summary>
        ///     Constructor.
        /// </summary>
        internal NeuroObject()
        {
            if (!infoShown)
            {
                Console.WriteLine("");
                Console.WriteLine("==================================================");
                Console.WriteLine("| XPIDEA.NEURO.NET.DLL                           |");
                Console.WriteLine("|                                                |");
                Console.WriteLine("| IS FREE FOR NON-COMMERCIAL USE                 |");
                Console.WriteLine("| This software must be purchased if it is to be |");
                Console.WriteLine("| used in any commercial endeavors, but may be   |");
                Console.WriteLine("| used free-of-charge for other purposes. Please |");
                Console.WriteLine("| refer to http://xpidea.com for more details.   |");
                Console.WriteLine("|                                                |");
                Console.WriteLine("| Copyright (C) 2001-2004 XPIdea.com             |");
                Console.WriteLine("| All rights reserved.                           |");
                Console.WriteLine("|                                                |");
                Console.WriteLine("| Author: Alex Cherkasov                         |");
                Console.WriteLine("| Email: support@xpidea.com                      |");
                Console.WriteLine("| WEB: http://xpidea.com                         |");
                Console.WriteLine("==================================================");
                Console.WriteLine("");
                infoShown = true;
            }
        }

        /// <summary>
        ///     Method used on epoch training of neural network. Must be executed every time
        ///     after each itteration through the pattern set. This method calculates average delta.
        /// </summary>
        /// <param name="epoch">Number of patterns that was presentend during this learning cycle.</param>
        public virtual void Epoch(int epoch)
        {
        }

        /// <summary>
        ///     Stores the object in a binary stream.
        /// </summary>
        /// <param name="binaryWriter">Binary stream writer.</param>
        public virtual void Save(BinaryWriter binaryWriter)
        {
        }

        /// <summary>
        ///     Restores object data from a binary stream.
        /// </summary>
        /// <param name="binaryReader">Binary stream reader.</param>
        public virtual void Load(BinaryReader binaryReader)
        {
        }

        /// <summary>
        ///     Saves object to a file in a binary format.
        /// </summary>
        /// <param name="fileName">File name.</param>
        public virtual void SaveToFile(string fileName)
        {
            Stream stream = new FileStream(fileName, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None);
            var binaryWriter = new BinaryWriter(stream);
            Save(binaryWriter);
            stream.Close();
        }

        /// <summary>
        ///     Loads object data from a file.
        /// </summary>
        /// <param name="fileName">File name.</param>
        public virtual void LoadFromFile(string fileName)
        {
            Stream stream = new FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.Read);
            var binaryReader = new BinaryReader(stream);
            Load(binaryReader);
            stream.Close();
        }

        /// <summary>
        ///     Rounds given value to a nearest integer.
        ///     For example:
        ///     RoundToNextInt(0.2) is 1
        ///     RoundToNextInt(-0.9) is -1
        ///     RoundToNextInt(-9) is -9
        /// </summary>
        /// <param name="value">is a value to be rounded.</param>
        /// <returns>Rounded to the nearest integer value of value</returns>
        public static int RoundToNextInt(double value)
        {
            var result = (int) Math.Round(value);
            if (value > 0)
            {
                if (value > result) result++;
            }
            else
            {
                if (value < result) result--;
            }
            return result;
        }

        /// <summary>
        ///     Returns random value between min and max (inclusive).
        /// </summary>
        /// <param name="min">First value.</param>
        /// <param name="max">Second value.</param>
        /// <returns> </returns>
        public static double Random(double min, double max)
        {
            double result, aRange;
            if (min > max)
            {
                result = max;
                max = min;
                min = result;
            }
            if (min == max) return max;
            aRange = max - min;
            return random.NextDouble()*aRange + min;
        }
    }

    /// <summary>
    ///     Implements an exception thrown on any execution error.
    /// </summary>
    public class ENeuroException : ApplicationException
    {
        /// <summary>
        ///     Cunstructor. Creates the exception.
        /// </summary>
        /// <param name="message">Error message.</param>
        public ENeuroException(string message) : base(message)
        {
        }
    }

    /// <summary>
    ///     Connects two nodes in a network.
    /// </summary>
    public class NeuroLink : NeuroObject
    {
        /// <summary>
        ///     Refers to source and destination node connected by this link.
        /// </summary>
        protected NeuroNode inNode, outNode;

        /// <summary>
        ///     Link weight.
        /// </summary>
        protected double linkWeight;

        /// <summary>
        ///     Constructor.
        /// </summary>
        public NeuroLink()
        {
            inNode = null;
            outNode = null;
        }

        /// <summary>
        ///     Property defines weight of the link.
        /// </summary>
        public double Weight
        {
            get { return GetLinkWeight(); }
            set { SetLinkWeight(value); }
        }

        /// <summary>
        ///     A source node. The node link is comming from.
        /// </summary>
        public NeuroNode InNode
        {
            get { return inNode; }
        }

        /// <summary>
        ///     A Destination node. A node the link is going to.
        /// </summary>
        public NeuroNode OutNode
        {
            get { return outNode; }
        }

        /// <summary>
        ///     Getter of <see cref="xpidea.neuro.net.NeuroLink.Weight" /> property.
        /// </summary>
        /// <returns>Link's weight.</returns>
        protected virtual double GetLinkWeight()
        {
            return linkWeight;
        }

        /// <summary>
        ///     Setter of <see cref="xpidea.neuro.net.NeuroLink.Weight" /> property.
        /// </summary>
        /// <param name="value">New link's weight value</param>
        protected virtual void SetLinkWeight(double value)
        {
            linkWeight = value;
        }

        /// <summary>
        ///     Overridden.Loads link data from a binary stream.
        /// </summary>
        /// <param name="binaryReader">Binary stream reader.</param>
        public override void Load(BinaryReader binaryReader)
        {
            base.Load(binaryReader);
            linkWeight = binaryReader.ReadDouble();
        }

        /// <summary>
        ///     Overridden.Stores link data in a binary stream.
        /// </summary>
        /// <param name="binaryWriter">Binary stream writer.</param>
        public override void Save(BinaryWriter binaryWriter)
        {
            base.Save(binaryWriter);
            binaryWriter.Write(linkWeight);
        }

        /// <summary>
        ///     Setter for <see cref="xpidea.neuro.net.NeuroLink.InNode" /> property.
        /// </summary>
        /// <param name="node">Incomming (source) node.</param>
        public void SetInNode(NeuroNode node)
        {
            inNode = node;
        }

        /// <summary>
        ///     Setter for <see cref="xpidea.neuro.net.NeuroLink.OutNode" /> property.
        /// </summary>
        /// <param name="node">Outgoing (destination) node.</param>
        public void SetOutNode(NeuroNode node)
        {
            outNode = node;
        }

        /// <summary>
        ///     Updates weight of the link on specified delta.
        /// </summary>
        /// <param name="deltaWeight">Weight change value.</param>
        public virtual void UpdateWeight(double deltaWeight)
        {
            Weight += deltaWeight;
        }

        /// <summary>
        ///     Weighted <see cref="xpidea.neuro.net.NeuroLink.InNode" /> value.
        /// </summary>
        /// <returns>InNode.Value * Weight</returns>
        public virtual double WeightedInValue()
        {
            return InNode.Value*Weight;
        }

        /// <summary>
        ///     Weighted <see cref="xpidea.neuro.net.NeuroLink.OutNode" /> value.
        /// </summary>
        /// <returns>OutNode.Value * Weight</returns>
        public virtual double WeightedOutValue()
        {
            return OutNode.Value*Weight;
        }

        /// <summary>
        ///     Weighted <see cref="xpidea.neuro.net.NeuroLink.InNode" /> error.
        /// </summary>
        /// <returns>InNode.Error * Weight</returns>
        public virtual double WeightedInError()
        {
            return InNode.Error*Weight;
        }

        /// <summary>
        ///     Weighted <see cref="xpidea.neuro.net.NeuroLink.OutNode" /> error.
        /// </summary>
        /// <returns>OutNode.Error * Weight</returns>
        public virtual double WeightedOutError()
        {
            return OutNode.Error*Weight;
        }
    }

    /// <summary>
    ///     A collection that stores <see cref='xpidea.neuro.net.NeuroLink' /> objects.
    /// </summary>
    /// <seealso cref='xpidea.neuro.net.NeuroLinkCollection' />
    [Serializable]
    public class NeuroLinkCollection : NeuroObjectCollection
    {
        /// <summary>
        ///     <para>
        ///         Initializes a new instance of <see cref='xpidea.neuro.net.NeuroLinkCollection' />.
        ///     </para>
        /// </summary>
        public NeuroLinkCollection()
        {
        }

        /// <summary>
        ///     <para>
        ///         Initializes a new instance of <see cref='xpidea.neuro.net.NeuroLinkCollection' /> based on another
        ///         <see cref='xpidea.neuro.net.NeuroLinkCollection' />.
        ///     </para>
        /// </summary>
        /// <param name='value'>
        ///     A <see cref='xpidea.neuro.net.NeuroLinkCollection' /> from which the contents are copied
        /// </param>
        public NeuroLinkCollection(NeuroLinkCollection value)
        {
            AddRange(value);
        }

        /// <summary>
        ///     <para>
        ///         Initializes a new instance of <see cref='xpidea.neuro.net.NeuroLinkCollection' /> containing any array of
        ///         <see cref='xpidea.neuro.net.NeuroLink' /> objects.
        ///     </para>
        /// </summary>
        /// <param name='value'>
        ///     A array of <see cref='xpidea.neuro.net.NeuroLink' /> objects with which to intialize the collection
        /// </param>
        public NeuroLinkCollection(NeuroLink[] value)
        {
            AddRange(value);
        }

        /// <summary>
        ///     <para>Represents the entry at the specified index of the <see cref='xpidea.neuro.net.NeuroLink' />.</para>
        /// </summary>
        /// <param name='index'>
        ///     <para>The zero-based index of the entry to locate in the collection.</para>
        /// </param>
        /// <value>
        ///     <para> The entry at the specified index of the collection.</para>
        /// </value>
        /// <exception cref='System.ArgumentOutOfRangeException'>
        ///     <paramref name='index' /> is outside the valid range of indexes
        ///     for the collection.
        /// </exception>
        public NeuroLink this[int index]
        {
            get { return ((NeuroLink) (List[index])); }
            set { List[index] = value; }
        }

        /// <summary>
        ///     Overridden.Creates new object contained by collection.
        /// </summary>
        /// <returns>NeuroObject</returns>
        protected override NeuroObject CreateContainigObject()
        {
            return new NeuroLink();
        }

        /// <summary>
        ///     <para>
        ///         Adds a <see cref='xpidea.neuro.net.NeuroLink' /> with the specified value to the
        ///         <see cref='xpidea.neuro.net.NeuroLinkCollection' /> .
        ///     </para>
        /// </summary>
        /// <param name='value'>The <see cref='xpidea.neuro.net.NeuroLink' /> to add.</param>
        /// <returns>
        ///     <para>The index at which the new element was inserted.</para>
        /// </returns>
        /// <seealso cref='xpidea.neuro.net.NeuroLinkCollection.AddRange' />
        public int Add(NeuroLink value)
        {
            return List.Add(value);
        }

        /// <summary>
        ///     <para>Copies the elements of an array to the end of the <see cref="xpidea.neuro.net.NeuroLinkCollection" />.</para>
        /// </summary>
        /// <param name="value">
        ///     An array of type <see cref="xpidea.neuro.net.NeuroLink" /> containing the objects to add to the collection.
        /// </param>
        /// <seealso cref="xpidea.neuro.net.NeuroLinkCollection.Add" />
        public void AddRange(NeuroLink[] value)
        {
            for (var i = 0; (i < value.Length); i = (i + 1))
            {
                Add(value[i]);
            }
        }

        /// <summary>
        ///     <para>
        ///         Adds the contents of another <see cref="xpidea.neuro.net.NeuroLinkCollection" /> to the end of the collection.
        ///     </para>
        /// </summary>
        /// <param name="value">
        ///     A <see cref="xpidea.neuro.net.NeuroLinkCollection" /> containing the objects to add to the collection.
        /// </param>
        /// <seealso cref="xpidea.neuro.net.NeuroLinkCollection.Add" />
        public void AddRange(NeuroLinkCollection value)
        {
            for (var i = 0; (i < value.Count); i = (i + 1))
            {
                Add(value[i]);
            }
        }

        /// <summary>
        ///     <para>
        ///         Gets a value indicating whether the
        ///         <see cref='xpidea.neuro.net.NeuroLinkCollection' /> contains the specified
        ///         <see cref='xpidea.neuro.net.NeuroLink' />.
        ///     </para>
        /// </summary>
        /// <param name='value'>The <see cref='xpidea.neuro.net.NeuroLink' /> to locate.</param>
        /// <returns>
        ///     <para>
        ///         <see langword='true' /> if the <see cref='xpidea.neuro.net.NeuroLink' /> is contained in the collection;
        ///         otherwise, <see langword='false' />.
        ///     </para>
        /// </returns>
        /// <seealso cref='xpidea.neuro.net.NeuroLinkCollection.IndexOf' />
        public bool Contains(NeuroLink value)
        {
            return List.Contains(value);
        }

        /// <summary>
        ///     <para>
        ///         Copies the <see cref="xpidea.neuro.net.NeuroLinkCollection" /> values to a one-dimensional
        ///         <see cref="System.Array" /> instance at the
        ///         specified index.
        ///     </para>
        /// </summary>
        /// <param name="array">
        ///     <para>
        ///         The one-dimensional <see cref="System.Array" /> that is the destination of the values copied from
        ///         <see cref="xpidea.neuro.net.NeuroLinkCollection" /> .
        ///     </para>
        /// </param>
        /// <param name="index">The index in <paramref name="array" /> where copying begins.</param>
        /// <exception cref="System.ArgumentException">
        ///     <para><paramref name="array" /> is multidimensional.</para>
        ///     <para>-or-</para>
        ///     <para>
        ///         The number of elements in the <see cref="xpidea.neuro.net.NeuroLinkCollection" /> is greater than the
        ///         available space between <paramref name="arrayIndex" /> and the end of <paramref name="array" />.
        ///     </para>
        /// </exception>
        /// <exception cref="System.ArgumentNullException"><paramref name="array" /> is <see langword="null" />. </exception>
        /// <exception cref="System.ArgumentOutOfRangeException">
        ///     <paramref name="arrayIndex" /> is less than
        ///     <paramref name="array" />'s lowbound.
        /// </exception>
        /// <seealso cref="System.Array" />
        public void CopyTo(NeuroLink[] array, int index)
        {
            List.CopyTo(array, index);
        }

        /// <summary>
        ///     <para>
        ///         Returns the index of a <see cref='xpidea.neuro.net.NeuroLink' /> in
        ///         the <see cref='xpidea.neuro.net.NeuroLinkCollection' /> .
        ///     </para>
        /// </summary>
        /// <param name='value'>The <see cref='xpidea.neuro.net.NeuroLink' /> to locate.</param>
        /// <returns>
        ///     <para>
        ///         The index of the <see cref='xpidea.neuro.net.NeuroLink' /> of <paramref name='value' /> in the
        ///         <see cref='xpidea.neuro.net.NeuroLinkCollection' />, if found; otherwise, -1.
        ///     </para>
        /// </returns>
        /// <seealso cref='xpidea.neuro.net.NeuroLinkCollection.Contains' />
        public int IndexOf(NeuroLink value)
        {
            return List.IndexOf(value);
        }

        /// <summary>
        ///     <para>
        ///         Inserts a <see cref="xpidea.neuro.net.NeuroLink" /> into the
        ///         <see cref="xpidea.neuro.net.NeuroLinkCollection" /> at the specified index.
        ///     </para>
        /// </summary>
        /// <param name="index">The zero-based index where <paramref name="value" /> should be inserted.</param>
        /// <param name=" value">The <see cref="xpidea.neuro.net.NeuroLink" /> to insert.</param>
        /// <seealso cref="xpidea.neuro.net.NeuroLinkCollection.Add" />
        public void Insert(int index, NeuroLink value)
        {
            List.Insert(index, value);
        }

        /// <summary>
        ///     <para>
        ///         Returns an enumerator that can iterate through
        ///         the <see cref='xpidea.neuro.net.NeuroLinkCollection' /> .
        ///     </para>
        /// </summary>
        /// <returns>
        ///     <para>None.</para>
        /// </returns>
        /// <seealso cref='System.Collections.IEnumerator' />
        public new CustomNeuroLinkEnumerator GetEnumerator()
        {
            return new CustomNeuroLinkEnumerator(this);
        }

        /// <summary>
        ///     <para>
        ///         Removes a specific <see cref="xpidea.neuro.net.NeuroLink" /> from the
        ///         <see cref="xpidea.neuro.net.NeuroLinkCollection" /> .
        ///     </para>
        /// </summary>
        /// <param name="value">
        ///     The <see cref="xpidea.neuro.net.NeuroLink" /> to remove from the
        ///     <see cref="xpidea.neuro.net.NeuroLinkCollection" /> .
        /// </param>
        /// <exception cref="System.ArgumentException"><paramref name="value" /> is not found in the Collection. </exception>
        public void Remove(NeuroLink value)
        {
            List.Remove(value);
        }

        /// <summary>
        ///     Cusom collection enumerator.
        /// </summary>
        public class CustomNeuroLinkEnumerator : object, IEnumerator
        {
            private readonly IEnumerator baseEnumerator;
            private readonly IEnumerable temp;

            /// <summary>
            ///     Constructor.
            /// </summary>
            /// <param name="mappings">Collection to be enumerated.</param>
            public CustomNeuroLinkEnumerator(NeuroLinkCollection mappings)
            {
                temp = mappings;
                baseEnumerator = temp.GetEnumerator();
            }

            /// <summary>
            ///     Points to Current element in collection.
            /// </summary>
            public NeuroLink Current
            {
                get { return ((NeuroLink) (baseEnumerator.Current)); }
            }

            object IEnumerator.Current
            {
                get { return baseEnumerator.Current; }
            }

            bool IEnumerator.MoveNext()
            {
                return baseEnumerator.MoveNext();
            }

            void IEnumerator.Reset()
            {
                baseEnumerator.Reset();
            }

            /// <summary>
            ///     Advances the enumerator to the next element of the collection.
            /// </summary>
            /// <returns>
            ///     true if the enumerator was successfully advanced to the next element; false if the enumerator has passed the
            ///     end of the collection.
            /// </returns>
            public bool MoveNext()
            {
                return baseEnumerator.MoveNext();
            }

            /// <summary>
            ///     Sets the enumerator to its initial position, which is before the first element in the collection.
            /// </summary>
            public void Reset()
            {
                baseEnumerator.Reset();
            }
        }
    }

    /// <summary>
    ///     Represents the Node in a neural network.
    /// </summary>
    public class NeuroNode : NeuroObject
    {
        /// <summary>
        ///     Incomming and Outgoing links of this node.
        /// </summary>
        private readonly NeuroLinkCollection outLinks;

        /// <summary>
        ///     Node value and node error.
        /// </summary>
        protected double nodeValue, nodeError;

        /// <summary>
        ///     Constructor.
        /// </summary>
        public NeuroNode()
        {
            InLinks = new NeuroLinkCollection();
            outLinks = new NeuroLinkCollection();
        }

        /// <summary>
        ///     Incomming links for this node.
        /// </summary>
        public NeuroLinkCollection InLinks { get; private set; }

        /// <summary>
        ///     Outgoing links of this node.
        /// </summary>
        public NeuroLinkCollection OutLinks
        {
            get { return outLinks; }
        }

        /// <summary>
        ///     <b>Value</b> property of the node.
        /// </summary>
        public double Value
        {
            get { return GetNodeValue(); }
            set { SetNodeValue(value); }
        }

        /// <summary>
        ///     <b>Error</b> property of the node.
        /// </summary>
        public double Error
        {
            get { return GetNodeError(); }
            set { SetNodeError(value); }
        }

        /// <summary>
        ///     Getter methos of <see cref="xpidea.neuro.net.NeuroNode.Value" /> property.
        /// </summary>
        /// <returns>Node value.</returns>
        protected virtual double GetNodeValue()
        {
            return nodeValue;
        }

        /// <summary>
        ///     Setter method of <see cref="xpidea.neuro.net.NeuroNode.Value" /> property.
        /// </summary>
        /// <param name="value">New node value.</param>
        protected virtual void SetNodeValue(double value)
        {
            nodeValue = value;
        }

        /// <summary>
        ///     Getter method of <see cref="xpidea.neuro.net.NeuroNode.Error" /> property.
        /// </summary>
        /// <returns>Node error.</returns>
        protected virtual double GetNodeError()
        {
            return nodeError;
        }

        /// <summary>
        ///     Setter method of <see cref="xpidea.neuro.net.NeuroNode.Error" /> property.
        /// </summary>
        /// <param name="error">Node error.</param>
        protected virtual void SetNodeError(double error)
        {
            nodeError = error;
        }

        /// <summary>
        ///     Executes node functionality.
        /// </summary>
        public virtual void Run()
        {
        }

        /// <summary>
        ///     Teaches the node.
        /// </summary>
        public virtual void Learn()
        {
        }

        /// <summary>
        ///     Connects this node to <b>toNode</b> using link <b>link</b>.
        /// </summary>
        /// <param name="toNode">Destination node.</param>
        /// <param name="link">Link used to connect nodes.</param>
        public void LinkTo(NeuroNode toNode, NeuroLink link)
        {
            OutLinks.Add(link);
            toNode.InLinks.Add(link);
            link.SetInNode(this);
            link.SetOutNode(toNode);
        }

        /// <summary>
        ///     Overridden.Loads node data from a binary stream.
        /// </summary>
        /// <param name="binaryReader">Binary stream reader.</param>
        public override void Load(BinaryReader binaryReader)
        {
            base.Load(binaryReader);
            nodeValue = binaryReader.ReadDouble();
            nodeError = binaryReader.ReadDouble();
        }

        /// <summary>
        ///     Overridden.Stores node data into binary stream.
        /// </summary>
        /// <param name="binaryWriter">Binary stream writer.</param>
        public override void Save(BinaryWriter binaryWriter)
        {
            base.Save(binaryWriter);
            binaryWriter.Write(nodeValue);
            binaryWriter.Write(nodeError);
        }
    }

    /// <summary>
    ///     Base abstract class for all neural networks.
    /// </summary>
    public abstract class NeuralNetwork : NeuroNode
    {
        /// <summary>
        ///     Links in the network.
        /// </summary>
        protected NeuroLink[] links;

        /// <summary>
        ///     Number of links in the network.
        /// </summary>
        protected int linksCount;

        /// <summary>
        ///     Nodes in the netowrk.
        /// </summary>
        protected NeuroNode[] nodes;

        /// <summary>
        ///     Number of nodes in the network.
        /// </summary>
        protected int nodesCount;

        /// <summary>
        ///     Constructor. Creates new instance of the network.
        /// </summary>
        public NeuralNetwork()
        {
            nodesCount = 0;
            linksCount = 0;
            nodes = null;
            links = null;
        }

        /// <summary>
        ///     Creates the network and loads it's state from a file.
        /// </summary>
        /// <param name="fileName">A file name.</param>
        public NeuralNetwork(string fileName)
        {
            nodesCount = 0;
            linksCount = 0;
            nodes = null;
            links = null;
            LoadFromFile(fileName);
        }

        /// <summary>
        ///     Returns network type. Used for persistent purposes.
        /// </summary>
        public NeuralNetworkType NetworkType
        {
            get { return GetNetworkType(); }
        }

        /// <summary>
        ///     Total number of nodes in the network.
        /// </summary>
        public int NodesCount
        {
            get { return nodesCount; }
        }

        /// <summary>
        ///     Total number of links in the network.
        /// </summary>
        public int LinksCount
        {
            get { return linksCount; }
        }

        /// <summary>
        ///     Number of input nodes.
        /// </summary>
        public int InputNodesCount
        {
            get { return GetInputNodesCount(); }
        }

        /// <summary>
        ///     Number of output nodes.
        /// </summary>
        public int OutputNodesCount
        {
            get { return GetOutPutNodesCount(); }
        }

        private void CheckNetworkType(BinaryReader binaryReader)
        {
            var nt = (NeuralNetworkType) binaryReader.ReadInt32();
            if (NetworkType != nt)
                throw new ENeuroException("Cannot load data. Invalid format.");
        }

        private void SaveNetworkType(BinaryWriter binaryWriter)
        {
            binaryWriter.Write((int) NetworkType);
        }

        /// <summary>
        ///     Performs network construction based on specific topology.  Connects all nodes in the network using the links.
        /// </summary>
        protected virtual void CreateNetwork()
        {
        }

        /// <summary>
        ///     Loads data into input nodes of the network.
        /// </summary>
        protected virtual void LoadInputs()
        {
        }

        /// <summary>
        ///     Neural network type.
        /// </summary>
        /// <returns>Type of neural network.</returns>
        protected abstract NeuralNetworkType GetNetworkType();

        /// <summary>
        ///     Returns number of input nodes in the network.
        /// </summary>
        /// <returns>Number of input nodes in the network.</returns>
        protected abstract int GetInputNodesCount();

        /// <summary>
        ///     Returns number of output nodes in the network.
        /// </summary>
        /// <returns>Number of output nodes in the network.</returns>
        protected abstract int GetOutPutNodesCount();

        /// <summary>
        ///     Returns N-th input node in the network.
        /// </summary>
        /// <param name="index">Node index.</param>
        /// <returns>Input node.</returns>
        protected abstract NeuroNode GetInputNode(int index);

        /// <summary>
        ///     Returns N-th output node in the network.
        /// </summary>
        /// <param name="index">Node index.</param>
        /// <returns>Output node.</returns>
        protected abstract NeuroNode GetOutputNode(int index);

        /// <summary>
        ///     Overridden.Finalizes trainig cycle of the network. Used by <see cref="xpidea.neuro.net.NeuralNetwork.Train" />
        ///     method of the network.
        /// </summary>
        /// <param name="epoch">Number of patterns was exposed to the network.</param>
        public override void Epoch(int epoch)
        {
            foreach (var node in nodes) node.Epoch(epoch);
            foreach (var link in links) link.Epoch(epoch);
            base.Epoch(epoch);
        }

        /// <summary>
        ///     Overridden.Loads network data from a binary stream.
        /// </summary>
        /// <param name="binaryReader">Binary stream reader.</param>
        public override void Load(BinaryReader binaryReader)
        {
            CheckNetworkType(binaryReader);
            nodesCount = binaryReader.ReadInt32();
            linksCount = binaryReader.ReadInt32();
            CreateNetwork();
            foreach (var node in nodes) node.Load(binaryReader);
            foreach (var link in links) link.Load(binaryReader);
        }

        /// <summary>
        ///     Overridden.Saves the network to a binary stream.
        /// </summary>
        /// <param name="binaryWriter">Binary stream writer.</param>
        public override void Save(BinaryWriter binaryWriter)
        {
            SaveNetworkType(binaryWriter);
            binaryWriter.Write(NodesCount);
            binaryWriter.Write(LinksCount);
            foreach (var node in nodes) node.Save(binaryWriter);
            foreach (var link in links) link.Save(binaryWriter);
        }

        /// <summary>
        ///     Performs network training. Here you write the code to train your network.
        /// </summary>
        /// <param name="patterns">Set of the patterns that will be exposed to a network during the training.</param>
        /// <remarks>
        ///     <p>
        ///         There are several major paradigms, or approaches, to neural network learning. These include
        ///         <i>supervised, unsupervised</i>, and <i>reinforcement</i> learning. How the training data is processed is a
        ///         major aspect of these learning paradigms.
        ///     </p>
        ///     <p>
        ///         <i>Supervised</i> learning is the most common form of learning and is sometimes called programming by example.
        ///         The neural network is trained by showing it examples of the problem state or attributes along with the desired
        ///         output or action. The neural network makes a prediction based on the inputs and if the output differs from
        ///         the desired out put, then the network is adjusted or adapted to produce the correct output. This process is
        ///         repeated over and over until the agent learns to make accurate classifications or predictions. Historical data
        ///         from databases, sensor logs, or trace logs is often used as the training or example data.
        ///     </p>
        ///     <p>
        ///         <i>Unsupervised</i> learning is used when the neural network needs to recognize similarities between inputs or
        ///         to identify features in the input data. The data is presented to the network, and it adapts so that it
        ///         partitions the data into groups. The clustering or segmenting process continues until the neural network places
        ///         the
        ///         same data into the same group on successive passes over the data. An unsupervised learning algorithm performs a
        ///         type of feature detection where important common attributes in the data are extracted. The Kohonen map will be
        ///         a good example of the network using unsupervised learning.
        ///     </p>
        ///     <p>
        ///         <i>Reinforcement</i> learning is a type of supervised learning used when explicit input/ output pairs of
        ///         training data are not available. It can be used in cases where there is a sequence of inputs arid the desired
        ///         output is only known after the specific sequence occurs. This process of identifying the relationship between a
        ///         series of input values and a later output value is called temporal credit assignment. Because we provide less
        ///         specific error information, reinforcement learning usually takes longer than supervised learning and is less
        ///         efficient. However, in many situations, having exact prior information about the desired outcome is not
        ///         possible. In many ways,
        ///         reinforcement learning is the most realistic form of learning.
        ///     </p>
        /// </remarks>
        public virtual void Train(PatternsCollection patterns)
        {
        }

        /// <summary>
        ///     Returns N-th input node of the network.<seealso cref="xpidea.neuro.net.NeuralNetwork.InputNodesCount" />
        /// </summary>
        /// <param name="index">Node index.</param>
        /// <returns>Node.</returns>
        public NeuroNode InputNode(int index)
        {
            return GetInputNode(index);
        }

        /// <summary>
        ///     Returns N-th output node of the network. <seealso cref="xpidea.neuro.net.NeuralNetwork.OutputNodesCount" />
        /// </summary>
        /// <param name="index">Node index</param>
        /// <returns>Node</returns>
        public NeuroNode OutputNode(int index)
        {
            return GetOutputNode(index);
        }
    }

    /// <summary>
    ///     A node that produce its value based on sum of values of it's inputs.
    /// </summary>
    public class FeedForwardNode : NeuroNode
    {
        /// <summary>
        ///     An Activation function of the node.
        /// </summary>
        /// <param name="value">Sum of input values of the node.</param>
        /// <returns>result</returns>
        /// <remarks>
        ///     Activation functions for the hidden nodes are needed to introduce
        ///     nonlinearity into the network. You can override this method to introduce your own function.
        /// </remarks>
        protected virtual double Transfer(double value)
        {
            return value;
        }

        /// <summary>
        ///     Overridden.Execute node's functionality.
        /// </summary>
        public override void Run()
        {
            double total = 0;
            foreach (var link in InLinks) total += link.WeightedInValue();
            Value = Transfer(total);
        }
    }

    /// <summary>
    ///     Base class for all input nodes.
    ///     Represents an input node to a network.
    /// </summary>
    public class InputNode : NeuroNode
    {
    }

    /// <summary>
    ///     Implements the node that always produces constant output value (bias).
    /// </summary>
    public class BiasNode : InputNode
    {
        /// <summary>
        ///     Constructs the Bias node.
        /// </summary>
        /// <param name="biasValue">Node value.</param>
        public BiasNode(double biasValue)
        {
            nodeValue = biasValue;
        }

        /// <summary>
        ///     Overridden.Overriden to prevent setting of node value directly.
        /// </summary>
        /// <param name="value">not used</param>
        protected override void SetNodeValue(double value)
        {
            //Cannot change value of BiasNode
        }
    }
}