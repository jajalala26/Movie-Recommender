'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''


import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        random_state = self.random_state # inherited RandomState object
        user_id_N, item_id_N, y_N = train_tuple  # Extract training data

        # TODO fix the lines below to have right dimensionality & values
        # TIP: use self.n_factors to access number of hidden dimensions


        self.param_dict = dict(
            mu=ag_np.full(1, 0.001, dtype=None),
            # mu=ag_np.array([ag_np.mean(y_N)]),
            # b_per_user=ag_np.zeros(n_users) + 0.0001, # FIX dimensionality
            # c_per_item=ag_np.zeros(n_items) + 0.0001, # FIX dimensionality
            b_per_user=ag_np.full(n_users, 0.001, dtype=None),
            c_per_item=ag_np.full(n_items, 0.001, dtype=None),
            U=0.01 * random_state.randn(n_users, self.n_factors), # FIX dimensionality
            V=0.01 * random_state.randn(n_items, self.n_factors), # FIX dimensionality
            )


    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
        if mu is None: mu = self.param_dict['mu']
        if b_per_user is None: b_per_user = self.param_dict['b_per_user']
        if c_per_item is None: c_per_item = self.param_dict['c_per_item']
        if U is None: U = self.param_dict['U']
        if V is None: V = self.param_dict['V']

        dot_product = ag_np.sum(U[user_id_N] * V[item_id_N], axis=1)
        yhat_N = mu + b_per_user[user_id_N] + c_per_item[item_id_N] + dot_product

        return yhat_N


    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''

        user_id_N = data_tuple[0]
        item_id_N = data_tuple[1]
        y_N = data_tuple[2]

        N = y_N.size
        yhat_N = self.predict(user_id_N, item_id_N, **param_dict)

        # calculate mean squared error for loss

        mse_loss = ag_np.mean((y_N - yhat_N) **2)
        abs_diff_sum = ag_np.sum(ag_np.abs(y_N - yhat_N))
        loss = abs_diff_sum / N

        reg_loss = self.alpha * (ag_np.sum(param_dict['b_per_user']**2) + ag_np.sum(param_dict['c_per_item']**2))

        return mse_loss + reg_loss

    def calculate_mae(self, data_tuple):
            """Calculate Mean Absolute Error (MAE) on given data tuple."""
            user_id_N = data_tuple[0]
            item_id_N = data_tuple[1]
            y_N = data_tuple[2]

            # Get predictions using current parameters
            yhat_N = self.predict(user_id_N, item_id_N)

            # Calculate the absolute difference and mean it
            mae = ag_np.mean(ag_np.abs(y_N - yhat_N))
            return mae

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, n_factors in enumerate([2, 10, 50]):
        model = CollabFilterOneVectorPerItem(
            n_epochs=100, batch_size=5000, step_size=5,
            n_factors=n_factors, alpha=0.0  # No regularization (alpha=0)
        )
        model.init_parameter_dict(n_users, n_items, train_tuple)

        model.fit(train_tuple, valid_tuple)

        train_mae = model.calculate_mae(train_tuple)
        valid_mae = model.calculate_mae(valid_tuple)
        test_mae = model.calculate_mae(test_tuple)
        print(f"n_factors = {n_factors}, Test MAE: {test_mae}")

        # Plot MAE for the current n_factors
        axes[idx].plot(model.trace_epoch, model.trace_mae_train, label="Training MAE")
        axes[idx].plot(model.trace_epoch, model.trace_mae_valid, label="Validation MAE")
        axes[idx].set_title(f"n_factors = {n_factors}")
        axes[idx].set_xlabel('Epochs')
        axes[idx].set_ylabel('Mean Absolute Error')
        axes[idx].legend()

    plt.tight_layout()
    plt.show()
