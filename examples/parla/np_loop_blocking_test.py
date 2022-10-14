"""
Simple unit test for blocking with numpy arrays 
"""

import numpy as np

def main():

    # Specify some array dimensions
    num_local = 4
    num_global = 20800 
    
    # Specify a partition of the (global) list of elements
    num_blocks = 4 # How many blocks do you want to use? 
    elements_per_block = num_global//num_blocks # Block size
    leftover_blocks = num_global % num_blocks
    
    # Adjust the number of elements if the block size doesn't divide the elements evenly
    block_sizes = elements_per_block*np.ones([num_blocks], dtype=np.int64)
    block_sizes[0:leftover_blocks] += 1
    
    print("Number of blocks: " + str(num_blocks))
    print("Number of left-over blocks: " + str(leftover_blocks))
    print("Elements per block:", block_sizes,"\n")

    # Create an array to hold the global data for the RHS
    global_array = np.zeros([num_global])

    # To use the blocking scheme, we'll
    # also use another set of arrays that
    # hold partial sums of this global array
    block_global_array = np.zeros([num_blocks, num_global])

    # Next we loop over each block which partitions the element indices
    # Each block will form a task
    for i in range(num_blocks):

        # Need the offset for the element indices owned by this block
        # This is the sum of all block sizes that came before it
        s_idx = np.sum(block_sizes[:i])
        e_idx = s_idx + block_sizes[i]

        # Next, loop over the block elements and mimic quadrature evaluations
        for j in range(s_idx, e_idx):

            # Pick "num_local" random locations in global_array where writes occur
            # This mimics the scatter step in the FEM when writing from local to global arrays
            #local_idx = np.random.choice(num_global, size=num_local, replace=False)
            local_idx = np.arange(0, 2*num_local, 2)

            # This local array would normally be obtained using quadrature
            # There are as many entries as local dof
            # We assume the integrals returned are equal to 1 for simplicity
            local_array = np.ones(num_local)

            # Accumulate the local array into the relevant entries of the block-wise global vector
            block_global_array[i,local_idx] += local_array[:]

    print("(Before the reduction across blocks) block_global_array[:,:10] = ", block_global_array[:,:10], "\n")

    # Perform the reduction across the blocks
    global_array = np.sum(block_global_array, axis=0)

    print("(After reduction across blocks) block_global_array[:,:10] = ", block_global_array[:,:10], "\n")

    print("global array[:10] = ", global_array[:10], "\n")



if __name__ == "__main__":

    main()






