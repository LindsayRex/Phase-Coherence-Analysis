**Background:**
Research suggests wavelets could be adapted for reconstructing wideband signals from narrowband chunks, addressing phase coherence 
through decomposition and alignment, but no direct examples exist for SDRs. The evidence leans toward the  idea being innovative, 
as most related work focuses on speech or other domains, not SDRs with limited bandwidth. This presents an opportunity for novel research,
potentially leveraging wavelet packet decomposition and inverse transforms to ensure phase coherence, though practical implementation 
would require addressing SDR-specific challenges like oscillator drift.

Here are three key ideas being reserched in this project to tackle the phase coherence problem when reconstructing a high-fidelity,
phase-coherent wideband signal from sequentially captured narrowband chunks using wavelets with cheaper SDRs, along with their reference mathematics:

**1. Wavelet Packet Decomposition with Phase Alignment**
Idea: Use wavelet packet decomposition (WPD) to break each narrowband chunk into time-frequency components, then align the phase across
chunks by estimating and correcting phase offsets during reconstruction. This leverages wavelets’ ability to provide detailed frequency
resolution, allowing phase continuity to be maintained across chunk boundaries.

How It Works: WPD extends the discrete wavelet transform (DWT) by decomposing both approximation and detail coefficients, offering finer
frequency granularity. After capturing narrowband chunks (e.g., 56 MHz segments of a 400 MHz signal), WPD analyzes each chunk.
Phase offsets due to SDR oscillator drift or timing jitter are estimated by comparing overlapping regions or using a reference signal. 
The inverse WPD then reconstructs the wideband signal, adjusting coefficients to ensure phase coherence.

**Reference Mathematics:**
Wavelet Packet Decomposition: For a signal ( x(t) ), the WPD at level ( j ) and node ( n ) is:
W_{j,n}(t) = \sum_k c_{j,n}(k) \psi_{j,k}(t)
where \psi_{j,k}(t) = 2^{j/2} \psi(2^j t - k) is the wavelet function, c_{j,n}(k) are coefficients, and ( n ) indexes the frequency subband.
Phase Correction: For two adjacent chunks with coefficients c_{j,n}^1 and c_{j,n}^2, estimate phase offset \Delta\phi as:
\Delta\phi = \arg\left(\sum_k c_{j,n}^1(k) \cdot (c_{j,n}^2(k))^*\right)
Adjust c_{j,n}^2 by multiplying with e^{-i\Delta\phi} before inverse transform.

Inverse WPD: Reconstruct via:
x(t) = \sum_{j,n} \sum_k c_{j,n}'(k) \psi_{j,k}(t)
where c_{j,n}' are phase-corrected coefficients.

Source: Adapted from “Radio frequency interference detection and mitigation” (Hindawi, 2014), which uses WPD for signal separation, extendable to phase alignment.

**2. Stationary Wavelet Transform (SWT) for Phase Preservation**
Idea: Apply the Stationary Wavelet Transform (SWT), which avoids downsampling (unlike DWT), to preserve phase information across all scales and time shifts.
 This ensures that phase relationships within and between narrowband chunks are maintained during decomposition and reconstruction.
How It Works: SWT decomposes each 56 MHz chunk without decimation, producing redundant coefficients that retain temporal and phase details.
By processing chunks sequentially and aligning their SWT coefficients (e.g., using cross-correlation or a pilot tone), the inverse SWT
reconstructs a phase-coherent wideband signal. This is particularly useful for non-stationary signals where phase varies over time.

Reference Mathematics:
SWT Decomposition: For a signal ( x(t) ), at level ( j ):
A_j(k) = \sum_m h(m) A_{j-1}(k + m), \quad D_j(k) = \sum_m g(m) A_{j-1}(k + m)
where A_j and D_j are approximation and detail coefficients, ( h ) and ( g ) are low-pass and high-pass filters, and no downsampling occurs (shift-invariant).
Phase Alignment: Align chunks by minimizing phase discontinuity in overlapping regions:
\Delta\phi = \arg\min_{\phi} \left\| A_j^1(k) - e^{i\phi} A_j^2(k) \right\|_2
Inverse SWT: Reconstruct via:
x(t) = A_J(t) + \sum_{j=1}^J D_j(t)
summing adjusted coefficients across levels ( J ).

Source: Inspired by “TFAW: Wavelet-based signal reconstruction” (A&A, 2018), which uses SWT for signal shape preservation, adaptable to phase coherence.

**3. Adaptive Wavelet Design with Phase-Locking**
Idea: Design a custom wavelet tailored to the signal’s characteristics (e.g., communication signals) and incorporate a phase-locking mechanism during reconstruction. This adaptive approach optimizes phase continuity by matching the wavelet to the signal’s frequency and phase structure, reducing errors at chunk boundaries.
How It Works: Construct a wavelet basis (e.g., via lifting schemes) that emphasizes phase-sensitive features of the target signal. Decompose each narrowband chunk, then use a phase-locking algorithm (e.g., based on a reference signal or cross-chunk correlation) to adjust coefficients before reconstruction. This could minimize phase discontinuities introduced by sequential SDR captures.

**Reference Mathematics:**
Adaptive Wavelet Construction: Define a mother wavelet \psi(t) via lifting:
\psi(t) = \sum_k a_k \phi(2t - k) + \sum_k b_k \psi(2t - k)
where \phi(t) is a scaling function, and coefficients a_k, b_k are tuned to signal properties (e.g., via optimization against a training signal).
Phase-Locking: For chunk coefficients c_{j,k}^1 and c_{j,k}^2, compute phase shift:
\Delta\phi = \frac{1}{N} \sum_{k \in \text{overlap}} \angle\left(\frac{c_{j,k}^1}{(c_{j,k}^2)^*}\right)
Adjust c_{j,k}^2 \leftarrow c_{j,k}^2 e^{-i\Delta\phi}.
Reconstruction: Use inverse DWT:
x(t) = \sum_j \sum_k c_{j,k}' \psi_{j,k}(t)
with phase-locked coefficients c_{j,k}'.

Source: Derived from “Critically-Sampled Wavelet Reconstruction” (MathWorks) and general wavelet theory, with phase-locking as a novel extension.


**How to Use:**

To Generate Simulated data signal to analysis: 
    
    /Phase-Coherence-Analysis/ ./PC_Simulated_data_generator.py

To reconstruct the 400MHz signal with the reconstruction module using wavelets: 

    /Phase-Coherence-Analysis/Python -m "Reconstruction Module.main"

  This will read the  .h5  file data and output the graphs and analysis accordingly.  

**Requirements: **

    CUDA / NVIDIA GPU will help a lot otherwise calculations will fall back to back to CPU.



