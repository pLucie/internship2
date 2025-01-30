import os
from keras.models import Model
from keras.layers import Input, Conv2D
from utils import get_completed_runs, log_completed_runs, load_completed_runs_from_log
from resunet import ResUNet

def train_and_save_models(output_dir, epochs_list, loss_functions, ground_truth_weights, input_image, groundtruth_image):
    """
    Automates model training and saving.
    """
    log_file_path = os.path.join(output_dir, "completed_runs.log")
    completed_runs = load_completed_runs_from_log(log_file_path)

    for epochs in epochs_list:
        for loss_name, loss_fn in loss_functions.items():
            for weight in ground_truth_weights:
                run_id = f"{epochs}epochs_{loss_name}_weight{weight:.1f}"

                if run_id in completed_runs:
                    print(f"Skipping already completed run: {run_id}")
                    continue

                print(f"Training with {epochs} epochs, loss: {loss_name}, weight: {weight}")

                # Model architecture
                input_layer = Input(shape=(2500, 2500, 13))
                resunet_features = ResUNet(input_layer)
                feature_output = Conv2D(5, (1, 1), activation='linear', name='feature_output')(resunet_features)
                classification_output = Conv2D(1, (1, 1), activation='sigmoid', name='classification_output')(feature_output)
                model = Model(inputs=input_layer, outputs=[feature_output, classification_output])

                # Compile model
                if loss_name == 'weighted_binary_crossentropy':
                    model.compile(
                        optimizer='adam',
                        loss={
                            'feature_output': None,
                            'classification_output': lambda y_true, y_pred: loss_fn(y_true, y_pred, weight_1=weight)
                        },
                        loss_weights={'feature_output': 0.0, 'classification_output': 1.0}
                    )
                elif loss_name == 'weighted_f05_loss':
                    model.compile(
                        optimizer='adam',
                        loss={
                            'feature_output': None,
                            'classification_output': loss_fn
                        },
                        loss_weights={'feature_output': 0.0, 'classification_output': 1.0}
                    )

                # Train model
                model.fit(
                    input_image,
                    [groundtruth_image, groundtruth_image],  # Same groundtruth for both outputs
                    epochs=epochs,
                    batch_size=1
                )

                # Save model and weights
                model_name = f"resunet_{epochs}epochs_{loss_name}_weight{weight:.1f}.h5"
                model_save_path = os.path.join(output_dir, model_name)
                model.save(model_save_path)
                print(f"Model saved to {model_save_path}")

                weights_name = f"resunet_{epochs}epochs_{loss_name}_weight{weight:.1f}_weights.h5"
                weights_save_path = os.path.join(output_dir, weights_name)
                model.save_weights(weights_save_path)
                print(f"Weights saved to {weights_save_path}")

                # Log completed run
                with open(log_file_path, "a") as log_file:
                    log_file.write(run_id + "\n")
