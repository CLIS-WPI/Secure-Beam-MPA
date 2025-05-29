# -*- coding: utf-8 -*-
"""
======================================================================================
 SINGLE-FILE SCRIPT (UPDATED VERSION) FOR:
 Proactive and Strategy-Aligned mmWave Security:
 A Predictive, Multimodal Beamforming Framework using Preference Optimization
======================================================================================
 
 
 VERSION HIGHLIGHTS:
 - Utilizes sionna.phy.mimo.grid_of_beams_dft for efficient and standard beam codebook generation.
 - Fully runnable, end-to-end implementation from data generation to evaluation.
 
 HOW TO RUN THIS SCRIPT:
 -----------------------
 This script operates in three distinct modes. You must run them in order.
 Open your terminal or command prompt and execute the following commands:
 
 1. Generate Training Data:
    python main.py --mode generate-data --num_samples 20000
 
 2. Train the Model (Pre-training + DPO Alignment):
    python main.py --mode train --epochs 20 --dpo_epochs 5
 
 3. Evaluate the Final Aligned Model:
    python main.py --mode evaluate
 
"""
import numpy as np
import tensorflow as tf
import os
import time
import pandas as pd
import logging
from tqdm import tqdm
import argparse

# --- SETUP AND CONFIGURATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

logging.basicConfig(filename='predictive_beamforming_v2.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from sionna.phy.channel.tr38901 import UMa
    from sionna.phy.channel.tr38901.antenna import PanelArray
    from sionna.phy.mimo import grid_of_beams_dft # UPDATED: Import the native beam codebook generator
except ImportError as e:
    logging.error(f"FATAL: Failed to import Sionna components. Please ensure Sionna is installed correctly. Error: {e}")
    print(f"FATAL ERROR: Sionna components could not be imported. Please check your installation. Details: {e}")
    exit()

np.random.seed(42)
tf.random.set_seed(42)

# --- CHANNEL SIMULATOR CLASS (UPDATED) ---
class MmWaveChannelSimulator:
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.carrier_frequency = 28e9
        self.bandwidth = 100e6
        
        # UPDATED: Use Sionna's native DFT beam codebook generation
        self.num_ant_v = 8 # Vertical antennas [cite: 1]
        self.num_ant_h = 8 # Horizontal antennas [cite: 1]
        oversampling = 2   # Oversampling factor for denser codebook [cite: 1]
        
        self.beam_codebook = grid_of_beams_dft(
            num_ant_v=self.num_ant_v,
            num_ant_h=self.num_ant_h,
            oversmpl_v=oversampling,
            oversmpl_h=oversampling
        ) # [cite: 1, 2]
        self.num_beams = self.beam_codebook.shape[0]
        logging.info(f"Generated DFT beam codebook with {self.num_beams} beams.")

        # Create a parallel array of angles for evaluation purposes
        # This gives the approximate horizontal angle for each beam in the codebook
        num_angles_h = self.num_ant_h * oversampling
        self.beam_codebook_angles_rad = np.arcsin(np.arange(-num_angles_h/2, num_angles_h/2) / (num_angles_h/2) * (1/np.sqrt(2)))
        # Repeat for the vertical grid dimension
        self.beam_codebook_angles_rad = np.tile(self.beam_codebook_angles_rad, self.num_ant_v * oversampling)

        self.bs_array = PanelArray(
            num_rows_per_panel=self.num_ant_v, 
            num_cols_per_panel=self.num_ant_h, 
            polarization="single",
            polarization_type="V", 
            antenna_pattern="38.901", 
            carrier_frequency=self.carrier_frequency
        )
        self.ut_array = PanelArray(
            num_rows_per_panel=1, 
            num_cols_per_panel=1, 
            polarization='single',
            polarization_type='V', 
            antenna_pattern='omni', 
            carrier_frequency=self.carrier_frequency
        )
        self.num_bs_antennas = self.bs_array.num_ant
        
        self.channel_model = UMa(
            carrier_frequency=self.carrier_frequency, 
            o2i_model="low", 
            ut_array=self.ut_array,
            bs_array=self.bs_array, 
            direction="downlink", 
            enable_pathloss=True, 
            enable_shadow_fading=True
        )
        
        self.tx_power_dbm = 30.0
        self.tx_power = 10**((self.tx_power_dbm - 30) / 10)
        self.noise_power_db_per_hz = -174.0
        no_db = self.noise_power_db_per_hz + 10 * np.log10(self.bandwidth)
        self.noise_power = 10**((no_db - 30) / 10)
        self.sensing_range_max = 150.0

        self.bs_position = tf.constant([[[0.0, 0.0, 10.0]]], dtype=tf.float32)
        self.user_position = tf.Variable([[[50.0, 10.0, 1.5]]], dtype=tf.float32)
        self.attacker_position = tf.Variable([[[40.0, -15.0, 1.5]]], dtype=tf.float32)

    def find_optimal_beam(self):
        h, _ = self.channel_model(self.batch_size, 1, self.bandwidth)
        h_user = tf.squeeze(h)

        sinrs = []
        # UPDATED: Iterate directly over the precoding vectors from the codebook
        for i in range(self.num_beams):
            precoder = self.beam_codebook[i] # Get the precoder for the i-th beam
            
            effective_channel = tf.reduce_sum(h_user * precoder)
            signal_power = tf.square(tf.abs(effective_channel)) * self.tx_power
            sinr = signal_power / self.noise_power
            sinrs.append(sinr.numpy())
            
        optimal_beam_index = np.argmax(sinrs)
        max_sinr = 10 * np.log10(np.max(sinrs) + 1e-12)
        return optimal_beam_index, max_sinr

    def generate_supervised_sample(self):
        # --- 1. Simulate Current State to Get Inputs ---
        self.channel_model.set_topology(ut_loc=self.user_position, bs_loc=self.bs_position)
        csi, _ = self.channel_model(self.batch_size, 1, self.bandwidth)
        csi = tf.squeeze(csi)

        bs_pos_np = self.bs_position.numpy().squeeze()
        attacker_pos_np = self.attacker_position.numpy().squeeze()
        true_attacker_vector = attacker_pos_np - bs_pos_np
        true_attacker_range = np.linalg.norm(true_attacker_vector)
        true_attacker_azimuth = np.arctan2(true_attacker_vector[1], true_attacker_vector[0])

        detected_az, detected_range, confidence = 0.0, 0.0, 0.0
        if 0 < true_attacker_range <= self.sensing_range_max:
            prob_detection = 0.8 * np.exp(-0.015 * true_attacker_range) 
            if np.random.rand() < prob_detection:
                confidence = prob_detection * (1 + np.random.normal(0, 0.1))
                detected_az = true_attacker_azimuth + np.random.normal(0, np.deg2rad(8))
                detected_range = true_attacker_range + np.random.normal(0, 4.0)
        
        isac_data = np.array([detected_az, detected_range, np.clip(confidence, 0, 1)], dtype=np.float32)

        # --- 2. Simulate Next State to Get Labels ---
        user_move = np.random.uniform(-0.5, 0.5, size=(1, 1, 3)) * np.array([[[1, 1, 0]]])
        attacker_move = np.random.uniform(-1.0, 1.0, size=(1, 1, 3)) * np.array([[[1, 1, 0]]])
        self.user_position.assign_add(user_move.astype(np.float32))
        self.attacker_position.assign_add(attacker_move.astype(np.float32))
        
        self.channel_model.set_topology(ut_loc=self.user_position, bs_loc=self.bs_position)
        
        optimal_next_beam_idx, max_sinr = self.find_optimal_beam()
        
        # --- 3. Package the Sample ---
        inputs = {'csi': csi.numpy(), 'context': isac_data}
        labels = {
            'location': self.user_position.numpy().squeeze(),
            'beam': tf.keras.utils.to_categorical(optimal_next_beam_idx, num_classes=self.num_beams)
        }
        supplemental_info = {
            'true_attacker_azimuth': true_attacker_azimuth,
            'max_possible_sinr': max_sinr
        }
        
        return inputs, labels, supplemental_info

# --- PREDICTIVE MODEL CLASS ---
class PredictiveBeamformer(tf.keras.Model):
    def __init__(self, num_beams, num_antennas):
        super().__init__()
        self.num_beams = num_beams
        self.num_antennas = num_antennas

        csi_input = tf.keras.layers.Input(shape=(self.num_antennas,), dtype=tf.complex64, name='csi_input')
        csi_real = tf.keras.layers.Lambda(lambda x: tf.math.real(x))(csi_input)
        csi_imag = tf.keras.layers.Lambda(lambda x: tf.math.imag(x))(csi_input)
        csi_proc = tf.keras.layers.Concatenate()([csi_real, csi_imag])
        csi_dense = tf.keras.layers.Dense(256, activation='relu')(csi_proc) # [cite: 2]

        context_input = tf.keras.layers.Input(shape=(3,), name='context_input')
        context_dense = tf.keras.layers.Dense(128, activation='relu')(context_input) # [cite: 2]
        
        fused = tf.keras.layers.Concatenate()([csi_dense, context_dense]) # [cite: 2]
        backbone = tf.keras.layers.Dense(256, activation='relu')(fused)
        backbone = tf.keras.layers.Dropout(0.3)(backbone)
        backbone = tf.keras.layers.Dense(128, activation='relu')(backbone)

        location_output = tf.keras.layers.Dense(3, name='location')(backbone) # [cite: 2]
        beam_output = tf.keras.layers.Dense(self.num_beams, activation='softmax', name='beam')(backbone) # [cite: 2]
        
        self.model = tf.keras.Model(
            inputs=[csi_input, context_input],
            outputs=[location_output, beam_output],
            name="PredictiveBeamformer"
        )

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

# --- DPO TRAINER CLASS ---
class DPO_Trainer:
    def __init__(self, policy_model, ref_model, beta=0.1):
        self.policy_model = policy_model
        self.ref_model = ref_model     
        self.beta = beta               

    def _get_log_probs(self, model, inputs, chosen_indices, rejected_indices):
        _, beam_probs = model(inputs, training=False)
        chosen_indices = tf.cast(chosen_indices, dtype=tf.int32)
        rejected_indices = tf.cast(rejected_indices, dtype=tf.int32)
        chosen_probs = tf.gather(beam_probs, chosen_indices, batch_dims=1)
        rejected_probs = tf.gather(beam_probs, rejected_indices, batch_dims=1)
        return tf.math.log(chosen_probs + 1e-9), tf.math.log(rejected_probs + 1e-9)

    @tf.function
    def train_step(self, optimizer, inputs, preference_data):
        chosen_beams = preference_data['chosen_beam_idx']
        rejected_beams = preference_data['rejected_beam_idx']
        
        with tf.GradientTape() as tape:
            pi_logps_chosen, pi_logps_rejected = self._get_log_probs(self.policy_model, inputs, chosen_beams, rejected_beams)
            ref_logps_chosen, ref_logps_rejected = self._get_log_probs(self.ref_model, inputs, chosen_beams, rejected_beams)
            log_ratio_chosen = pi_logps_chosen - ref_logps_chosen
            log_ratio_rejected = pi_logps_rejected - ref_logps_rejected
            loss = -tf.nn.log_sigmoid(self.beta * (log_ratio_chosen - log_ratio_rejected))
            final_loss = tf.reduce_mean(loss)

        gradients = tape.gradient(final_loss, self.policy_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.policy_model.trainable_variables))
        return final_loss

# --- MAIN EXECUTION LOGIC ---
def main(args):
    simulator = MmWaveChannelSimulator()
    model = PredictiveBeamformer(num_beams=simulator.num_beams, num_antennas=simulator.num_bs_antennas)

    if args.mode == 'generate-data':
        print(f"--- Mode: Generate Data ---")
        logging.info(f"Starting data generation for {args.num_samples} samples.")
        all_inputs = {'csi': [], 'context': []}
        all_labels = {'location': [], 'beam': []}
        for _ in tqdm(range(args.num_samples), desc="Generating Samples"):
            inputs, labels, _ = simulator.generate_supervised_sample()
            all_inputs['csi'].append(inputs['csi'])
            all_inputs['context'].append(inputs['context'])
            all_labels['location'].append(labels['location'])
            all_labels['beam'].append(labels['beam'])
        for key in all_inputs: all_inputs[key] = np.array(all_inputs[key])
        for key in all_labels: all_labels[key] = np.array(all_labels[key])
        np.savez('training_data.npz', **all_inputs, **all_labels)
        print(f"\nData generation complete. Saved {args.num_samples} samples to 'training_data.npz'")
        logging.info("Data generation complete.")

    elif args.mode == 'train':
        print(f"--- Mode: Train Model ---")
        logging.info("Starting training process.")
        
        # Phase 1: Supervised Pre-training [cite: 3]
        print("\n--- Phase 1: Supervised Pre-training ---")
        logging.info("Loading data for pre-training.")
        try:
            data = np.load('training_data.npz')
            inputs = [data['csi'], data['context']]
            labels = [data['location'], data['beam']]
        except FileNotFoundError:
            print("ERROR: 'training_data.npz' not found. Please run in 'generate-data' mode first.")
            logging.error("Training failed: training_data.npz not found.")
            return

        model.model.compile(optimizer='adam', 
                            loss={'location': 'mean_squared_error', 'beam': 'categorical_crossentropy'},
                            metrics={'location': 'mae', 'beam': 'accuracy'})
        
        print("Starting model.fit for pre-training...")
        model.model.fit(inputs, labels, epochs=args.epochs, batch_size=64, validation_split=0.2)
        model.save_weights('pretrained_model.weights.h5')
        print("\nPre-training complete. Weights saved to 'pretrained_model.weights.h5'")
        logging.info("Pre-training phase complete.")
        
        # Phase 2: DPO Fine-tuning [cite: 3]
        print("\n--- Phase 2: DPO Fine-tuning for Strategic Alignment ---")
        ref_model = PredictiveBeamformer(num_beams=simulator.num_beams, num_antennas=simulator.num_bs_antennas)
        _ = ref_model(inputs) 
        ref_model.load_weights('pretrained_model.weights.h5')
        ref_model.trainable = False

        print("Creating preference dataset...")
        logging.info("Creating DPO preference dataset.")
        preference_inputs = {'csi': [], 'context': []}
        preference_data = {'chosen_beam_idx': [], 'rejected_beam_idx': []}
        
        _, predicted_beam_probs = model.predict(inputs)
        predicted_beam_indices = np.argmax(predicted_beam_probs, axis=1)
        true_beam_indices = np.argmax(labels[1], axis=1)

        for i in tqdm(range(len(true_beam_indices)), desc="Creating Preferences"):
            if true_beam_indices[i] != predicted_beam_indices[i]:
                preference_inputs['csi'].append(inputs[0][i])
                preference_inputs['context'].append(inputs[1][i])
                preference_data['chosen_beam_idx'].append(true_beam_indices[i])
                preference_data['rejected_beam_idx'].append(predicted_beam_indices[i])

        for key in preference_inputs: preference_inputs[key] = np.array(preference_inputs[key])
        for key in preference_data: preference_data[key] = np.array(preference_data[key])
        
        if len(preference_data['chosen_beam_idx']) == 0:
            print("No preference pairs created. Skipping DPO.")
            logging.warning("DPO skipped as no preference pairs were generated.")
            model.save_weights('aligned_model.weights.h5')
            return
            
        dpo_dataset = tf.data.Dataset.from_tensor_slices((
            (preference_inputs['csi'], preference_inputs['context']),
            preference_data
        )).shuffle(10000).batch(32)
        
        print(f"Created {len(preference_data['chosen_beam_idx'])} preference pairs for DPO.")
        logging.info(f"DPO training with {len(preference_data['chosen_beam_idx'])} pairs.")

        dpo_trainer = DPO_Trainer(model, ref_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        
        for epoch in range(args.dpo_epochs):
            total_loss = 0
            for step, (batch_inputs, batch_prefs) in tqdm(enumerate(dpo_dataset), desc=f"DPO Epoch {epoch+1}/{args.dpo_epochs}", total=len(dpo_dataset)):
                loss = dpo_trainer.train_step(optimizer, batch_inputs, batch_prefs)
                total_loss += loss
            avg_loss = total_loss / (step + 1)
            print(f"DPO Epoch {epoch+1}/{args.dpo_epochs} complete. Average Loss: {avg_loss:.4f}")
            logging.info(f"DPO Epoch {epoch+1} loss: {avg_loss:.4f}")

        model.save_weights('aligned_model.weights.h5')
        print("\nDPO fine-tuning complete. Aligned model weights saved to 'aligned_model.weights.h5'")
        logging.info("Training process complete.")

    elif args.mode == 'evaluate':
        print("--- Mode: Evaluate Model ---")
        logging.info("Starting evaluation phase.")
        try:
            model.load_weights('aligned_model.weights.h5')
            print("Successfully loaded 'aligned_model.weights.h5' for evaluation.")
        except Exception as e:
            print(f"ERROR: Could not load 'aligned_model.weights.h5'. Please run 'train' mode first. Details: {e}")
            logging.error(f"Evaluation failed: Could not load aligned model weights. {e}")
            return

        num_eval_steps = 200
        results = []
        
        for step in tqdm(range(num_eval_steps), desc="Running Evaluation"):
            inputs, labels, sup_info = simulator.generate_supervised_sample()
            
            _, beam_prob_pred = model.predict([np.expand_dims(inputs['csi'],0), np.expand_dims(inputs['context'],0)], verbose=0)
            predicted_beam_idx = np.argmax(beam_prob_pred)
            
            true_next_beam_idx = np.argmax(labels['beam'])
            
            attacker_azimuth = sup_info['true_attacker_azimuth']
            predicted_beam_angle = simulator.beam_codebook_angles_rad[predicted_beam_idx]
            angle_diff = abs(predicted_beam_angle - attacker_azimuth)
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
            is_safe = 1 if angle_diff > np.deg2rad(20) else 0

            results.append({
                'Step': step,
                'Predicted_Beam_Idx': predicted_beam_idx,
                'True_Optimal_Beam_Idx': true_next_beam_idx,
                'Is_Correct (Accuracy)': 1 if predicted_beam_idx == true_next_beam_idx else 0,
                'Is_Safe (Strategic Goal)': is_safe,
                'Max_Possible_SINR_dB': sup_info['max_possible_sinr']
            })
            
        df_eval = pd.DataFrame(results)
        df_eval.to_csv('evaluation_results_predictive_v2.csv', index=False)
        
        print("\n--- Evaluation Summary ---")
        accuracy = df_eval['Is_Correct (Accuracy)'].mean() * 100
        safety_rate = df_eval['Is_Safe (Strategic Goal)'].mean() * 100
        
        print(f"Beam Prediction Accuracy: {accuracy:.2f}%")
        print(f"Strategic Safety Rate (Avoids Attacker): {safety_rate:.2f}%")
        print("\nEvaluation complete. Results saved to 'evaluation_results_predictive_v2.csv'")
        logging.info(f"Evaluation complete. Accuracy: {accuracy:.2f}%, Safety: {safety_rate:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Proactive and Aligned mmWave Beamforming Simulation")
    parser.add_argument('--mode', type=str, required=True, choices=['generate-data', 'train', 'evaluate'],
                        help="The mode to run the script in: 'generate-data', 'train', or 'evaluate'.")
    parser.add_argument('--num_samples', type=int, default=20000, 
                        help="Number of data samples to generate for training.")
    parser.add_argument('--epochs', type=int, default=20, 
                        help="Number of epochs for the initial supervised pre-training phase.")
    parser.add_argument('--dpo_epochs', type=int, default=5, 
                        help="Number of epochs for DPO fine-tuning.")
    
    args = parser.parse_args()
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s). Enabling memory growth.")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error during GPU setup: {e}")
    else:
        print("No GPU detected. Running on CPU.")
        
    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f"\nTotal execution time for mode '{args.mode}': {(end_time - start_time):.2f} seconds.")
