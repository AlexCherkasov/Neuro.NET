using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;
using xpidea.neuro.net.patterns;
using xpidea.neuro.net.backprop;


namespace xpidea.neuro.net.examples.backprop.ocr
{
	
	/// <summary>
	/// Summary description for Form1.
	/// </summary>
	public class Form1 : System.Windows.Forms.Form
	{
		static public bool IsTerminated;
		public class OCRNetwork: BackPropagationRPROPNetwork
		{
			
			private Form1 owner;
			public OCRNetwork (Form1 owner, int[] nodesInEachLayer):base(nodesInEachLayer)
			{
				this.owner = owner;
			}
			private int OutputPatternIndex(Pattern pattern)
			{
				for (int i = 0; i<pattern.OutputsCount;i++)
					if (pattern.Output[i] == 1 )
						return i;
				return -1;
			}

			public void AddNoiseToInputPattern(int levelPercent)
			{
				int i = ((NodesInLayer(0) - 1) * levelPercent)/100;
				while (i > 0) 
				{
					nodes[(int)(BackPropagationNetwork.Random(0, NodesInLayer(0) - 1))].Value = BackPropagationNetwork.Random(0, 100);
					i--;
				}

			}

			public int BestNodeIndex 
			{ 
				get 
				{
					int result = -1;
					double aMaxNodeValue = 0;
					double aMinError = double.PositiveInfinity;
					for (int i = 0; i< this.OutputNodesCount;i++)
					{
						NeuroNode node = OutputNode(i);
						if ((node.Value > aMaxNodeValue)||((node.Value >= aMaxNodeValue)&&(node.Error < aMinError))) 
						{
							aMaxNodeValue = node.Value;
							aMinError = node.Error;
							result = i;
						}
    
					}
					return result;
				}
			}
			public override void Train(PatternsCollection patterns) 
			{
							
				int  iteration = 0;
				if (patterns != null) 
				{
					double error = 0;
					int good = 0;
					while (good < patterns.Count) // Train until all patterns are correct
					{
						if (Form1.IsTerminated) return;
						error = 0;
						owner.progressBar1.Value = good;
						owner.label16.Text = "Training progress: " + ((good * 100)/owner.progressBar1.Maximum).ToString() + "%";
						good = 0;
						for (int i = 0; i<patterns.Count; i++)
						{
							for (int k = 0; k<NodesInLayer(0); k++)	
								nodes[k].Value = patterns[i].Input[k];
							AddNoiseToInputPattern(owner.trackBar3.Value);
							this.Run();
							for (int k = 0;k< this.OutputNodesCount;k++) 
							{
								error += Math.Abs(this.OutputNode(k).Error);
								this.OutputNode(k).Error = patterns[i].Output[k];
							}
							this.Learn();
							if (BestNodeIndex == OutputPatternIndex(patterns[i]))
								good++;
							
							iteration ++;						
							Application.DoEvents();

						}

						foreach (NeuroLink link in links) ((EpochBackPropagationLink)link).Epoch(patterns.Count);

						if ((iteration%2) == 0)
							owner.label17.Text = "AVG Error: " + (error / OutputNodesCount).ToString() + "  Iteration: " + iteration.ToString();
					}
					owner.label17.Text = "AVG Error: " + (error / OutputNodesCount).ToString() + "  Iteration: " + iteration.ToString();
				}

			}
		
		}
	
		public static int aMatrixDim = 10;
		public static byte aFirstChar =  (byte)'A';
		public static byte aLastChar = (byte)'z';
		public static int aCharsCount = aLastChar - aFirstChar +1;
		public PatternsCollection trainingPatterns;
		public OCRNetwork backpropNetwork;

		#region Variables
		private System.Windows.Forms.PictureBox pictureBox1;
		private System.Windows.Forms.TabControl tabControl1;
		private System.Windows.Forms.TabPage tabPage1;
		private System.Windows.Forms.TabPage tabPage2;
		private System.Windows.Forms.TabPage tabPage3;
		private System.Windows.Forms.TabPage tabPage4;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.Button button1;
		private System.Windows.Forms.Button button2;
		private System.Windows.Forms.Button button3;
		private System.Windows.Forms.Button button4;
		private System.Windows.Forms.Button button5;
		private System.Windows.Forms.TrackBar trackBar3;
		private System.Windows.Forms.Label label8;
		private System.Windows.Forms.Label label9;
		private System.Windows.Forms.Label label10;
		private System.Windows.Forms.TrackBar trackBar4;
		private System.Windows.Forms.Label label11;
		private System.Windows.Forms.Label label12;
		private System.Windows.Forms.Label label13;
		private System.Windows.Forms.Label label14;
		private System.Windows.Forms.Label label15;
		private System.Windows.Forms.ProgressBar progressBar1;
		private System.Windows.Forms.Label label16;
		private System.Windows.Forms.Label label17;
		private System.Windows.Forms.OpenFileDialog openFileDialog1;
		private System.Windows.Forms.SaveFileDialog saveFileDialog1;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		#endregion
		#region OtherJunk
		public Form1()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			label5.Text = "";
			for (int i=0; i<aCharsCount; i++)
				label5.Text += Convert.ToChar(aFirstChar + i) + " ";
		}

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		protected override void Dispose( bool disposing )
		{
			if( disposing )
			{
				if (components != null) 
				{
					components.Dispose();
				}
			}
			base.Dispose( disposing );
		}

		#region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.pictureBox1 = new System.Windows.Forms.PictureBox();
			this.tabControl1 = new System.Windows.Forms.TabControl();
			this.tabPage1 = new System.Windows.Forms.TabPage();
			this.button1 = new System.Windows.Forms.Button();
			this.label5 = new System.Windows.Forms.Label();
			this.label1 = new System.Windows.Forms.Label();
			this.tabPage2 = new System.Windows.Forms.TabPage();
			this.button2 = new System.Windows.Forms.Button();
			this.label2 = new System.Windows.Forms.Label();
			this.tabPage3 = new System.Windows.Forms.TabPage();
			this.label17 = new System.Windows.Forms.Label();
			this.progressBar1 = new System.Windows.Forms.ProgressBar();
			this.label9 = new System.Windows.Forms.Label();
			this.label8 = new System.Windows.Forms.Label();
			this.trackBar3 = new System.Windows.Forms.TrackBar();
			this.button5 = new System.Windows.Forms.Button();
			this.button4 = new System.Windows.Forms.Button();
			this.button3 = new System.Windows.Forms.Button();
			this.label3 = new System.Windows.Forms.Label();
			this.tabPage4 = new System.Windows.Forms.TabPage();
			this.label16 = new System.Windows.Forms.Label();
			this.label15 = new System.Windows.Forms.Label();
			this.label14 = new System.Windows.Forms.Label();
			this.label13 = new System.Windows.Forms.Label();
			this.label12 = new System.Windows.Forms.Label();
			this.label11 = new System.Windows.Forms.Label();
			this.label10 = new System.Windows.Forms.Label();
			this.trackBar4 = new System.Windows.Forms.TrackBar();
			this.label4 = new System.Windows.Forms.Label();
			this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
			this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
			this.tabControl1.SuspendLayout();
			this.tabPage1.SuspendLayout();
			this.tabPage2.SuspendLayout();
			this.tabPage3.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.trackBar3)).BeginInit();
			this.tabPage4.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.trackBar4)).BeginInit();
			this.SuspendLayout();
			// 
			// pictureBox1
			// 
			this.pictureBox1.Dock = System.Windows.Forms.DockStyle.Right;
			this.pictureBox1.Location = new System.Drawing.Point(392, 0);
			this.pictureBox1.Name = "pictureBox1";
			this.pictureBox1.Size = new System.Drawing.Size(200, 266);
			this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
			this.pictureBox1.TabIndex = 0;
			this.pictureBox1.TabStop = false;
			// 
			// tabControl1
			// 
			this.tabControl1.Controls.Add(this.tabPage1);
			this.tabControl1.Controls.Add(this.tabPage3);
			this.tabControl1.Controls.Add(this.tabPage2);
			this.tabControl1.Controls.Add(this.tabPage4);
			this.tabControl1.Dock = System.Windows.Forms.DockStyle.Fill;
			this.tabControl1.Location = new System.Drawing.Point(0, 0);
			this.tabControl1.Name = "tabControl1";
			this.tabControl1.SelectedIndex = 0;
			this.tabControl1.Size = new System.Drawing.Size(392, 266);
			this.tabControl1.TabIndex = 1;
			this.tabControl1.KeyPress += new System.Windows.Forms.KeyPressEventHandler(this.tabControl1_KeyPress);
			// 
			// tabPage1
			// 
			this.tabPage1.Controls.Add(this.button1);
			this.tabPage1.Controls.Add(this.label5);
			this.tabPage1.Controls.Add(this.label1);
			this.tabPage1.Location = new System.Drawing.Point(4, 22);
			this.tabPage1.Name = "tabPage1";
			this.tabPage1.Size = new System.Drawing.Size(384, 240);
			this.tabPage1.TabIndex = 0;
			this.tabPage1.Text = "Step 1";
			// 
			// button1
			// 
			this.button1.Location = new System.Drawing.Point(112, 192);
			this.button1.Name = "button1";
			this.button1.Size = new System.Drawing.Size(152, 24);
			this.button1.TabIndex = 2;
			this.button1.Text = "Generate training patterns";
			this.button1.Click += new System.EventHandler(this.button1_Click);
			// 
			// label5
			// 
			this.label5.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.label5.Location = new System.Drawing.Point(16, 48);
			this.label5.Name = "label5";
			this.label5.Size = new System.Drawing.Size(352, 104);
			this.label5.TabIndex = 1;
			this.label5.Text = "label5";
			this.label5.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
			// 
			// label1
			// 
			this.label1.BackColor = System.Drawing.SystemColors.InactiveCaption;
			this.label1.Dock = System.Windows.Forms.DockStyle.Top;
			this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.label1.ForeColor = System.Drawing.SystemColors.Window;
			this.label1.Location = new System.Drawing.Point(0, 0);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(384, 32);
			this.label1.TabIndex = 0;
			this.label1.Text = "   Step1:   Generate neural network training patterns";
			this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// tabPage2
			// 
			this.tabPage2.Controls.Add(this.button2);
			this.tabPage2.Controls.Add(this.label2);
			this.tabPage2.Location = new System.Drawing.Point(4, 22);
			this.tabPage2.Name = "tabPage2";
			this.tabPage2.Size = new System.Drawing.Size(384, 240);
			this.tabPage2.TabIndex = 1;
			this.tabPage2.Text = "Step 2";
			// 
			// button2
			// 
			this.button2.Location = new System.Drawing.Point(88, 128);
			this.button2.Name = "button2";
			this.button2.Size = new System.Drawing.Size(200, 24);
			this.button2.TabIndex = 6;
			this.button2.Text = "Create the network";
			this.button2.Click += new System.EventHandler(this.button2_Click);
			// 
			// label2
			// 
			this.label2.BackColor = System.Drawing.SystemColors.InactiveCaption;
			this.label2.Dock = System.Windows.Forms.DockStyle.Top;
			this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.label2.ForeColor = System.Drawing.SystemColors.Window;
			this.label2.Location = new System.Drawing.Point(0, 0);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(384, 32);
			this.label2.TabIndex = 1;
			this.label2.Text = "   Step2:  Create Backpropagation Neural Network";
			this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// tabPage3
			// 
			this.tabPage3.Controls.Add(this.label17);
			this.tabPage3.Controls.Add(this.progressBar1);
			this.tabPage3.Controls.Add(this.label9);
			this.tabPage3.Controls.Add(this.label8);
			this.tabPage3.Controls.Add(this.trackBar3);
			this.tabPage3.Controls.Add(this.button5);
			this.tabPage3.Controls.Add(this.button4);
			this.tabPage3.Controls.Add(this.button3);
			this.tabPage3.Controls.Add(this.label3);
			this.tabPage3.Location = new System.Drawing.Point(4, 22);
			this.tabPage3.Name = "tabPage3";
			this.tabPage3.Size = new System.Drawing.Size(384, 240);
			this.tabPage3.TabIndex = 2;
			this.tabPage3.Text = "Step 3";
			// 
			// label17
			// 
			this.label17.Dock = System.Windows.Forms.DockStyle.Bottom;
			this.label17.Location = new System.Drawing.Point(0, 200);
			this.label17.Name = "label17";
			this.label17.Size = new System.Drawing.Size(384, 16);
			this.label17.TabIndex = 9;
			this.label17.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
			// 
			// progressBar1
			// 
			this.progressBar1.Dock = System.Windows.Forms.DockStyle.Bottom;
			this.progressBar1.Location = new System.Drawing.Point(0, 216);
			this.progressBar1.Name = "progressBar1";
			this.progressBar1.Size = new System.Drawing.Size(384, 24);
			this.progressBar1.TabIndex = 8;
			// 
			// label9
			// 
			this.label9.Location = new System.Drawing.Point(184, 120);
			this.label9.Name = "label9";
			this.label9.Size = new System.Drawing.Size(176, 64);
			this.label9.TabIndex = 7;
			// 
			// label8
			// 
			this.label8.Location = new System.Drawing.Point(184, 40);
			this.label8.Name = "label8";
			this.label8.Size = new System.Drawing.Size(184, 23);
			this.label8.TabIndex = 6;
			this.label8.Text = "Add noise to the patterns";
			// 
			// trackBar3
			// 
			this.trackBar3.Location = new System.Drawing.Point(184, 64);
			this.trackBar3.Maximum = 100;
			this.trackBar3.Name = "trackBar3";
			this.trackBar3.Size = new System.Drawing.Size(176, 45);
			this.trackBar3.TabIndex = 5;
			this.trackBar3.TickFrequency = 5;
			this.trackBar3.TickStyle = System.Windows.Forms.TickStyle.Both;
			this.trackBar3.Scroll += new System.EventHandler(this.trackBar3_Scroll);
			// 
			// button5
			// 
			this.button5.Location = new System.Drawing.Point(16, 120);
			this.button5.Name = "button5";
			this.button5.Size = new System.Drawing.Size(120, 24);
			this.button5.TabIndex = 4;
			this.button5.Text = "Load trained network";
			this.button5.Click += new System.EventHandler(this.button5_Click);
			// 
			// button4
			// 
			this.button4.Location = new System.Drawing.Point(16, 88);
			this.button4.Name = "button4";
			this.button4.Size = new System.Drawing.Size(120, 24);
			this.button4.TabIndex = 3;
			this.button4.Text = "Save trained network";
			this.button4.Click += new System.EventHandler(this.button4_Click);
			// 
			// button3
			// 
			this.button3.Location = new System.Drawing.Point(16, 56);
			this.button3.Name = "button3";
			this.button3.Size = new System.Drawing.Size(120, 24);
			this.button3.TabIndex = 2;
			this.button3.Text = "Train the network";
			this.button3.Click += new System.EventHandler(this.button3_Click);
			// 
			// label3
			// 
			this.label3.BackColor = System.Drawing.SystemColors.InactiveCaption;
			this.label3.Dock = System.Windows.Forms.DockStyle.Top;
			this.label3.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.label3.ForeColor = System.Drawing.SystemColors.Window;
			this.label3.Location = new System.Drawing.Point(0, 0);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(384, 32);
			this.label3.TabIndex = 1;
			this.label3.Text = "   Step3:  Train the network";
			this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// tabPage4
			// 
			this.tabPage4.Controls.Add(this.label16);
			this.tabPage4.Controls.Add(this.label15);
			this.tabPage4.Controls.Add(this.label14);
			this.tabPage4.Controls.Add(this.label13);
			this.tabPage4.Controls.Add(this.label12);
			this.tabPage4.Controls.Add(this.label11);
			this.tabPage4.Controls.Add(this.label10);
			this.tabPage4.Controls.Add(this.trackBar4);
			this.tabPage4.Controls.Add(this.label4);
			this.tabPage4.Location = new System.Drawing.Point(4, 22);
			this.tabPage4.Name = "tabPage4";
			this.tabPage4.Size = new System.Drawing.Size(384, 240);
			this.tabPage4.TabIndex = 3;
			this.tabPage4.Text = "Step 4";
			// 
			// label16
			// 
			this.label16.Dock = System.Windows.Forms.DockStyle.Top;
			this.label16.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.label16.Location = new System.Drawing.Point(0, 32);
			this.label16.Name = "label16";
			this.label16.Size = new System.Drawing.Size(384, 16);
			this.label16.TabIndex = 14;
			this.label16.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
			// 
			// label15
			// 
			this.label15.Font = new System.Drawing.Font("Microsoft Sans Serif", 26.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.label15.Location = new System.Drawing.Point(264, 88);
			this.label15.Name = "label15";
			this.label15.Size = new System.Drawing.Size(40, 48);
			this.label15.TabIndex = 13;
			this.label15.Text = "A";
			// 
			// label14
			// 
			this.label14.BackColor = System.Drawing.SystemColors.Window;
			this.label14.Font = new System.Drawing.Font("Microsoft Sans Serif", 26.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.label14.Location = new System.Drawing.Point(80, 88);
			this.label14.Name = "label14";
			this.label14.Size = new System.Drawing.Size(40, 48);
			this.label14.TabIndex = 12;
			this.label14.Text = "A";
			// 
			// label13
			// 
			this.label13.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.label13.Location = new System.Drawing.Point(168, 104);
			this.label13.Name = "label13";
			this.label13.Size = new System.Drawing.Size(44, 16);
			this.label13.TabIndex = 11;
			this.label13.Text = "------>";
			// 
			// label12
			// 
			this.label12.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.label12.Location = new System.Drawing.Point(24, 56);
			this.label12.Name = "label12";
			this.label12.Size = new System.Drawing.Size(328, 16);
			this.label12.TabIndex = 10;
			this.label12.Text = "You\'ve just entered                            Recognized";
			// 
			// label11
			// 
			this.label11.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.label11.Location = new System.Drawing.Point(112, 208);
			this.label11.Name = "label11";
			this.label11.Size = new System.Drawing.Size(168, 16);
			this.label11.TabIndex = 9;
			this.label11.Text = "Please enter a character";
			// 
			// label10
			// 
			this.label10.Location = new System.Drawing.Point(16, 160);
			this.label10.Name = "label10";
			this.label10.Size = new System.Drawing.Size(128, 16);
			this.label10.TabIndex = 8;
			this.label10.Text = "Noise level (%)";
			this.label10.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
			// 
			// trackBar4
			// 
			this.trackBar4.Location = new System.Drawing.Point(144, 152);
			this.trackBar4.Maximum = 100;
			this.trackBar4.Name = "trackBar4";
			this.trackBar4.Size = new System.Drawing.Size(176, 45);
			this.trackBar4.TabIndex = 7;
			this.trackBar4.TickFrequency = 5;
			this.trackBar4.TickStyle = System.Windows.Forms.TickStyle.Both;
			this.trackBar4.Scroll += new System.EventHandler(this.trackBar4_Scroll);
			// 
			// label4
			// 
			this.label4.BackColor = System.Drawing.SystemColors.InactiveCaption;
			this.label4.Dock = System.Windows.Forms.DockStyle.Top;
			this.label4.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.label4.ForeColor = System.Drawing.SystemColors.Window;
			this.label4.Location = new System.Drawing.Point(0, 0);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(384, 32);
			this.label4.TabIndex = 1;
			this.label4.Text = "   Step4:  Testing";
			this.label4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// openFileDialog1
			// 
			this.openFileDialog1.DefaultExt = "neuro";
			this.openFileDialog1.FileName = "OCRNetwork";
			this.openFileDialog1.Filter = "Neural network (*.neuro)|*.neuro";
			this.openFileDialog1.Title = "Load neural network";
			// 
			// saveFileDialog1
			// 
			this.saveFileDialog1.DefaultExt = "neuro";
			this.saveFileDialog1.FileName = "OCRNetwork";
			this.saveFileDialog1.Filter = "Neural network (*.neuro)|*.neuro";
			this.saveFileDialog1.Title = "Store the network";
			// 
			// Form1
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(592, 266);
			this.Controls.Add(this.tabControl1);
			this.Controls.Add(this.pictureBox1);
			this.Name = "Form1";
			this.Text = "XPidea  -=Simple OCR Demo (www.xpidea.com)=-";
			this.Closing += new System.ComponentModel.CancelEventHandler(this.Form1_Closing);
			this.tabControl1.ResumeLayout(false);
			this.tabPage1.ResumeLayout(false);
			this.tabPage2.ResumeLayout(false);
			this.tabPage3.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize)(this.trackBar3)).EndInit();
			this.tabPage4.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize)(this.trackBar4)).EndInit();
			this.ResumeLayout(false);

		}
		#endregion

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main() 
		{
			Application.Run(new Form1());
		}
		#endregion

		public PatternsCollection CreateTrainingPatterns(Font font)
		{			 
			PatternsCollection result = new PatternsCollection(aCharsCount, aMatrixDim * aMatrixDim, aCharsCount);
			for (int i= 0; i<aCharsCount; i++)
			{
				double[] aBitMatrix = CharToDoubleArray(Convert.ToChar(aFirstChar + i), font, aMatrixDim, 0);
				for (int j = 0; j<aMatrixDim * aMatrixDim; j++) 
					result[i].Input[j] = aBitMatrix[j];
				result[i].Output[i] = 1;				
			}
			return result;
		}

		private void ShowNoise(Size sz, Graphics g,int noisePercent)
		{			
			int range =  sz.Height*sz.Width*noisePercent/200;			
			for (int i=0; i<range; i++)
			{
				int x = (int)OCRNetwork.Random(0,sz.Width);
				int y = (int)OCRNetwork.Random(0,sz.Height);
				Rectangle r = new Rectangle(x,y,0,0);				
				r.Inflate(1,1);
				Brush b;
				if ((OCRNetwork.Random(0,100))>80) //80% is black noise, 20% is white noise
					b = new SolidBrush(Color.White);
				else
					b = new SolidBrush(Color.Black);

				g.FillRectangle(b,r);
				b.Dispose();				
			}
			
		}

		public double[] CharToDoubleArray(char aChar, Font aFont, int aArrayDim, int aAddNoisePercent)
		{
			double[] result = new double[aArrayDim*aArrayDim];
			Graphics gr = label5.CreateGraphics();
			Size size =Size.Round(gr.MeasureString(aChar.ToString(),aFont));
			Bitmap aSrc = new Bitmap(size.Width,size.Height);
			Graphics bmp = Graphics.FromImage(aSrc);
			bmp.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.None;
			bmp.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
			bmp.Clear(Color.White);
			bmp.DrawString(aChar.ToString(),aFont,new SolidBrush(Color.Black),new Point(0,0),new StringFormat());
			ShowNoise(size,bmp,aAddNoisePercent);
			pictureBox1.Image = aSrc;
			Application.DoEvents();
			double xStep = (double)aSrc.Width/(double)aArrayDim;
			double yStep = (double)aSrc.Height/(double)aArrayDim;
			for (int i=0; i<aSrc.Width; i++)
				for (int j=0;j<aSrc.Height;j++)
				{
					int x = (int)((i/xStep));
					int y = (int)(j/yStep);
					Color c = aSrc.GetPixel(i,j);
					result[y*x+y]+=Math.Sqrt(c.R*c.R+c.B*c.B+c.G*c.G); //Convert to BW, I guess I can use B component of Alpha color space too...
				}
			return  Scale(result);
		}
		private double MaxOf(double[] src)
		{
			double res=double.NegativeInfinity;
			foreach (double d in src)
				if (d>res) res = d;
			return res;
		}

		private double[] Scale(double[] src)
		{
			double max = MaxOf(src);
			if (max!=0)
			{
				for(int i=0; i<src.Length; i++)
					src[i] = src[i]/max;
			}
			return src;					
		}
			 
		private void button1_Click(object sender, System.EventArgs e)
		{
			trainingPatterns = CreateTrainingPatterns(label5.Font);
			tabControl1.SelectedTab = tabPage2;
		}

		private void button2_Click(object sender, System.EventArgs e)
		{
			backpropNetwork = new OCRNetwork(this, new int[3] {aMatrixDim * aMatrixDim, (aMatrixDim * aMatrixDim + aCharsCount)/2, aCharsCount});
			tabControl1.SelectedTab = tabPage3;
		}

		private void button3_Click(object sender, System.EventArgs e)
		{
			if (backpropNetwork == null) 
			{
				MessageBox.Show("Please create the network first!");
				return;
			}
			if (trainingPatterns == null)
			{
				MessageBox.Show("Please create training patterns on STEP 1");
				return;
			}
			progressBar1.Maximum = aCharsCount;
			label9.Text = "While the network is training you can proceed to the STEP 4 and observe how the recognition quality progress during the training.";
			backpropNetwork.Train(trainingPatterns);
			MessageBox.Show("Network training successfully complete!");

			
		}
		
		

		private void tabControl1_KeyPress(object sender, System.Windows.Forms.KeyPressEventArgs e)
		{
			if (tabControl1.SelectedTab == tabPage4)
			{
				if (backpropNetwork == null)
				{
					MessageBox.Show("Please go to STEP 2 and create the network.");
					return;
				}
				label14.Text = e.KeyChar.ToString();				
				double[] aInput = CharToDoubleArray(e.KeyChar, label5.Font, aMatrixDim, trackBar4.Value);
				for (int i = 0; i< backpropNetwork.InputNodesCount;i++)
					backpropNetwork.InputNode(i).Value = aInput[i];
				backpropNetwork.Run();
				label15.Text= Convert.ToChar(aFirstChar + backpropNetwork.BestNodeIndex).ToString();	
			}
		}

		private void button4_Click(object sender, System.EventArgs e)
		{
			if (backpropNetwork == null)
			{
				MessageBox.Show("Network is not yet created.");
				return;
			}
			if (saveFileDialog1.ShowDialog()== DialogResult.OK)
				backpropNetwork.SaveToFile(saveFileDialog1.FileName);
		}

		private void button5_Click(object sender, System.EventArgs e)
		{
			if (backpropNetwork == null)
			{
				MessageBox.Show("Network is not yet created.");
				return;
			}
			if (openFileDialog1.ShowDialog()== DialogResult.OK)
				backpropNetwork.LoadFromFile(openFileDialog1.FileName);
		}

		private void Form1_Closing(object sender, System.ComponentModel.CancelEventArgs e)
		{
			Form1.IsTerminated=true;
		}

		private void trackBar3_Scroll(object sender, System.EventArgs e)
		{
		
			this.label8.Text = "Add noise to the patterns  ( "+trackBar3.Value.ToString()+"% )";
		}

		
		private void trackBar4_Scroll(object sender, System.EventArgs e)
		{
			this.label10.Text = "Noise level ( "+trackBar4.Value.ToString()+"% )";
		}

	
	}
}
