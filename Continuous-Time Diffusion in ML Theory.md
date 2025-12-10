# Schedule
1. Introduction, Brownian motion and Itô's calculus, basic concepts of SDE
2. Fokker-Planck equation, diffusion processes, convergence of diffusion process, coupling and mixing time
3. Functional inequalities and diffusion process convergence
4. Discretization of diffusion process, application to MCMC and stochastic optimization
5. Advanced topics in discretization and sampling, Score-based diffusion generative models
6. Denoising diffusion generative models, discretization and learning of diffusion models
7. RL basics, value functions and value learning
8. Continuous-time control problem, HJB equations, continuous-time RL
9. Advanced topics in continuous-time RL, applications to diffusion model fine-tuning

# Lecture 1. Introduction, Brownian motion and Itô's calculus, basic concepts of SDE

## Brownian Motion
**Motivation**: simple random walk
For a discrete-time stochastic process $X_{n+1}= X_{n} + Z_{n+1}$, where $X_{0}=0$ and $Z_n =\begin{cases}1, & \text{w.p. } \frac{1}{2} \\-1, & \text{w.p. } \frac{1}{2}\end{cases}$
Consider $X_{nt}$. By CLT, for fixed $t > 0$, $\frac{1}{\sqrt{n}}X_{\lfloor nt \rfloor} \xrightarrow{d} \mathcal{N}(0, t)$
Hope: $\left(\frac{1}{\sqrt{n}} X_{\lfloor nt \rfloor}: 0 \le t \le T\right) \xrightarrow{d} \text{Something}$
Let $\left(B_{t}: t\ge 0\right)$ be the limit. Equivalently, we can define BM as a Gaussian Process.
- For any $t_{1}$, $t_{2}$, $\cdots$, $t_{n}$, $B(t_{1})$, $B(t_{2})$, $\cdots$, $B(t_{n})$ is a Gaussian r.v.
- Independent increment: $(B(t)-B(s))\perp B(s)$ for $t \gt s$.
- $\mathbb{E}\left[B_{t}B_{s}\right]=min(t, s)$.
- Regularity: w.p.1, $\left(B_{t}: t \ge 0\right)$ is a continuous function. 
	- Everywhere continuous, nowhere differentiable.
	- Graph has fractal dimension 1.5
- BM is a martingale(鞅).
	- Definition of martingale: $\mathbb{E}\left[B_{t} \mid \left(B_{r}:0\le r \le s \right)\right] = B_{s}$
	- Important theorem of martingale: **Optimal stopping theorem**
		- ($T$ is stopping time, satisfying $\left|B_{T}\right|\le C \lt +\infty$), then $\mathbb{E}\left[B_{T}\right]= \mathbb{E}\left[B_0\right]=0$
It can be shown that processes satisfying these 5 properties.

## Itô's Calculus

>[!problem]
>Can we make sense of $\int^{t}_{0} Y_{s} \mathrm{d}B_{s}$ (for some adapted stochastic process $Y$), where $Y_{t}\in \mathcal{F}_t$
>- $\mathcal{F}_{t}$ is filtration (Information about $B$ within time $\left[0, t\right]$)

Recall Riemann-Stieltjes integral. 
Let $h \in \mathbb{C}^{1}$, $\int^{t}_{0}f(s) \mathrm{d}h(s) \coloneqq \lim\limits_{n\to +\infty} \sum\limits_{i=0}^{n} f(\xi_{i}) \left(h(s_{i+1}) - h(s_{i})\right)$
For BM, the limit of the sum depends on how we choose $\xi_{i} \in \left[s_{i}, s_{i+1}\right]$
**e.g.** For $\int^{1}_{0}B_{t}\mathrm{d}B_{t}$
- By selecting left end-point: $I_{l}(n) = \sum\limits_{i=0}^{n-1} B(s_{i})\left(B(s_{i+1})-B(s_{i})\right)$
- By selecting right end-point: $I_{r}(n) = \sum\limits_{i=0}^{n-1} B(s_{i+1})\left(B(s_{i+1})-B(s_{i})\right)$
$I_{r}(n) - I_{l}(n) = \sum\limits_{i=0}^{n-1}\left(B(s_{i+1})-B(s_{i})\right)^{2}$. Each term is square of $\mathcal{N}(0, s_{i+1}-s_{i})$, so $\mathbb{E}\left[I_{r}(n)\right]-\mathbb{E}\left[I_{l}(n)\right] = \sum\limits_{i=0}^{n-1}(s_{i+1}-s_{i})=1$
In fact, we can compute:
- $I_{l}(n) \rightarrow \frac{1}{2}(B_{1}^{2}-1)$
- $I_{r}(n) \rightarrow \frac{1}{2}(B_{1}^{2}+1)$

>[!definition]
>**Itô's calculus**. Always take left end-point to calculate the limit.
>Itô's calculus define $\int^{t}_{0}Y_{s}\mathrm{d}B_{s}$ as the limit of $\sum\limits_{i=0}^{n-1}Y_{t_{i}} (B_{t_{i+1}} - B_{t_{i}})$ ($\mathbb{L}^{2}$ convergence)
>- $\sum\limits_{i=0}^{n-1}Y_{t_{i}} (B_{t_{i+1}} - B_{t_{i}})$ is Cauchy in $\mathbb{L}^{2}$ metric

### Itô's formula
>[!problem]
>Is there analogue of Newton-Leibniz for Itô's calculus?
>- For Stratonovich integral. ✅
>- For Itô integral. Need correction

**e.g.** $\int^{1}_{0}B_{t}\mathrm{d}B_{t} \neq \frac{1}{2}B_{1}^{2} - \frac{1}{2}B_{0}^{2}$
Derivation: for $f \in \mathbb{C}^{2}$, $f(B_{t})-f(B_{0})=\sum\limits_{i=0}^{n-1} (f(B_{t_{i+1}})-f(B_{t_{i}}))$
Taylor expansion: $$f(B_{t_{i+1}})-f(B_{t_{i}}) = (B_{t_{i+1}}-B_{t_{i}})f^{\prime}(B_{t_{i}}) + \frac{1}{2}(B_{t_{i+1}}-B_{t_{i}})^{2} f^{\prime\prime}(B_{t_{i}}) + o(\left|B_{t_{i+1}}-B_{t_{i}}\right|^2)$$
Then, $$f(B_{t})-f(B_{0})= 
\sum\limits_{i=0}^{n-1} (B_{t_{i+1}}-B_{t_{i}})f^{\prime}(B_{t_{i}}) + 
\frac{1}{2} \sum\limits_{i=0}^{n-1} (B_{t_{i+1}}-B_{t_{i}})^{2} f^{\prime\prime}(B_{t_{i}}) + 
o(\sum\limits_{i=0}^{n-1} \left|B_{t_{i+1}}-B_{t_{i}}\right|^2)$$
- Term 1: $\int^{t}_{0}f^{\prime}(B_{s})\mathrm{d}B_{s}$.
- Term 2: $\mathbb{E}\left[ (B_{t_{i+1}}-B_{t_{i}})^{2} \right] = t_{i+1} - t_{i}$, so $\sum\limits_{i=0}^{n-1} (B_{t_{i+1}}-B_{t_{i}})^{2} f^{\prime\prime}(B_{t_{i}}) \to \int^{t}_{0}f^{\prime\prime}(B_{s})\mathrm{d}s$
- Term 3: $(B_{t_{i+1}}-B_{t_{i}})^{2} \sim t_{i+1} - t_{i}$, so this term converge to zero.

>[!theorem] Itô's formulae
>Integral form:
>$$\color{red}\boxed{f(B_{t})-f(B_{0}) = \int^{t}_{0} f^{\prime}(B_{s})\mathrm{d}B_{s} + \frac{1}{2} \int^{t}_{0} f^{\prime\prime}(B_{s})\mathrm{d}s}$$
Differential form:
$$\color{red}\boxed{\mathrm{d}f(B_{t}) = f^{\prime}(B_{t})\mathrm{d}B_{t} + \frac{1}{2} f^{\prime\prime}(B_{t})\mathrm{d}t}$$


**e.g.** $f(x) = x^{2}$
**Sol**. We have $B_{t}^{2} = B_{t}^{2}-B_{0}^{2} = 2\int^{t}_{0}B_{s}\mathrm{d}B_{s}+\frac{1}{2}\int^{t}_{0}2\mathrm{d}s$. Then, $\int^{t}_{0}B_{s}\mathrm{d}B_{s}=\frac{1}{2}(B_{t}^{2}-t)$
### Chain rule 
Suppose: $\mathrm{d}X_{t} = Y_{t}\mathrm{d}t + Z_{t}\mathrm{d}B_{t}$
Recall the chain rule in calculus, suppose $\mathrm{d}f(t) = g(t)\mathrm{d}t$, then $\mathrm{d}h\left(f(t)\right) = h^{\prime}\left(f(t)\right)g(t)\mathrm{d}t$
Calculate $\mathrm{d}f(X_{t}) = ?$
Idea: telescope sum, for each term:
$f(X_{t_{i+1}})-f(X_{t_{i}}) = (X_{t_{i+1}}-X_{t_{i}})f^{\prime}(X_{t_{i}}) + \frac{1}{2}(X_{t_{i+1}}-X_{t_{i}})^{2} f^{\prime\prime}(X_{t_{i}}) + o(\left|X_{t_{i+1}}-X_{t_{i}}\right|^2)$
- Term 1 $\approx f^{\prime}\left(X_{t_{i}})[Y_{t_{i}}(t_{i+1}-t_{i}) + Z_{t_{i}}(B_{t_{i+1}} - B_{t_{i}})\right]$
- Term 2 $\approx \frac{1}{2}f^{\prime\prime}(X_{t_{i}}) [Y_{t_{i}}(t_{i+1}-t_{i}) + Z_{t_{i}}(B_{t_{i+1}} - B_{t_{i}})]^{2} \approx \frac{1}{2}f^{\prime\prime}(X_{t_i})Z_{t_{i}}^{2}(B_{t_{i+1}} - B_{t_{i}})^{2}$
- Term 3: high order, converge to zero after summation.
It can be shown:
$$\sum\limits_{i=0}^{n-1} f^{\prime\prime}(X_{t_i})Z_{t_{i}}^{2}(B_{t_{i+1}} - B_{t_{i}})^{2} \to \int_{0}^{t}f^{\prime\prime}(X_{s})Z_{s}^{2}\mathrm{d}s$$
So, we have chain rule in Itô's calculus:
$$
f(X_{t})-f(X_{0}) = 
\int_{0}^{t}f^{\prime}(X_{s})\mathrm{d}X_{s} + \frac{1}{2}\int_{0}^{t}f^{\prime\prime}(X_{s})Z_{s}^{2}\mathrm{d}s
$$
And differential form is:
$$\mathrm{d}f(X_t)=f^{\prime}(X_{t})\mathrm{d}X_{t} + \frac{1}{2}f^{\prime\prime}(X_{t})Z_{t}^{2}\mathrm{d}t$$

Indeed, we define:
$$\begin{aligned}
\langle X\rangle_{t} &\coloneqq \lim\limits_{n\to +\infty}\sum\limits_{i=0}^{n-1}(X_{t_{i+1}}-X_{t_{i}})^{2} \\
&= \int_{0}^{t}Z_{s}^{2}\mathrm{d}s
\end{aligned}$$
Then, $\mathrm{d}\langle X \rangle_{t}=Z_{t}^{2}\mathrm{d}t$.
So, 
$$\color{red}\boxed{\mathrm{d}f(X_{t}) = f^{\prime}(X_{t})\mathrm{d}X_{t} + \frac{1}{2} f^{\prime\prime}(X_{t})\mathrm{d}\langle X \rangle_{t}}$$
**Extension**
Non-time homogeneous:
$$\mathrm{d}f(t, X_{t}) = \frac{\partial f}{\partial t}(t, X_{t})\mathrm{d}t + \frac{\partial f}{\partial X}(t, X_{t})\mathrm{d}X_{t} + \frac{1}{2} \frac{\partial^{2} f}{\partial X^{2}}(t, X_{t}) \mathrm{d}\langle X\rangle_{t}$$
Suppose that $\mathbf{X}_{t} = \mathbf{Y}_{t}\mathrm{d}t + \mathbf{Z}_{t}\mathrm{d}\mathbf{B}_{t}$, where $\mathbf{X}_{t}, \mathbf{Y}_{t} \in \mathbb{R}^{d}$, $\mathbf{Z}_{t} \in \mathbb{R}^{d \times d}$ and $(\mathbf{B}_{t}:t\ge 0)$ is d-dim BM.
$$\begin{align}
\mathrm{d}f(\mathbf{X}_{t}) &= \langle \nabla f(\mathbf{X}_{t}), \mathbf{Y}_{t} \rangle \mathrm{d}t + \langle \nabla f(\mathbf{X}_{t}), \mathbf{Z}_{t} \mathrm{d}\mathbf{B}_{t} \rangle + 
\color{red}{\frac{1}{2} \mathrm{Tr}(\mathbf{Z}_{t}^{\top}\nabla^{2}f(\mathbf{X}_{t})\mathbf{Z}_{t}) \mathrm{d}t}\\
&= \langle \nabla f(\mathbf{X}_{t}), \mathbf{Y}_{t} \rangle \mathrm{d}t + \langle \nabla f(\mathbf{X}_{t}), \mathbf{Z}_{t} \mathrm{d}\mathbf{B}_{t} \rangle + 
\color{red}{\frac{1}{2} \mathrm{Tr}(\nabla^{2}f(\mathbf{X}_{t})\mathbf{Z}_{t}\mathbf{Z}_{t}^{\top}) \mathrm{d}t}
\end{align}$$
> Consider the 2nd order term in Taylor expansion $\frac{1}{2} \left(\mathbf{Z}_{t_{i}}\left( \mathbf{B}_{t_{i+1}} - \mathbf{B}_{t_{i}} \right)\right)^{\top} \nabla^{2}f(\mathbf{X}_{t_{i}}) \left(\mathbf{Z}_{t_{i}}\left( \mathbf{B}_{t_{i+1}} - \mathbf{B}_{t_{i}} \right)\right)$

## Stochastic Differential Equation and Diffusion Process
Diffusion process is defined as: $\mathrm{d}\mathbf{X}_{t} = \mathbf{b}_{t}(\mathbf{X}_{t})\mathrm{d}t + \mathbf{\Sigma}_{t}(\mathbf{X}_{t})^{\frac{1}{2}}\mathrm{d}\mathbf{B}_{t}$
- Is a Markov process.
- Is a solution to an SDE. (Strong solution exists and is unique when $(\mathbf{b}, \mathbf{\Sigma})$ are Lipschitz)

**e.g.** **Ornstein-Uhlenbeck process**. $\mathrm{d}\mathbf{X}_{t} = -\mathbf{X}_{t}\mathrm{d}t + \mathrm{d}\mathbf{B}_{t}$. (In general, $\mathrm{d}\mathbf{X}_{t} = \mathbf{A}_{t} \mathbf{X}_{t}\mathrm{d}t + \mathbf{M}_{t} \mathrm{d}\mathbf{B}_{t}$, where $\mathbf{A}_t$, $\mathbf{M}_{t} \in \mathbb{R}^{d\times d}$)
Note that $\mathrm{d}(e^{t}\mathbf{X}_{t}) = e^{t}\mathbf{X}_{t}\mathrm{d}t + e^{t} \mathrm{d}\mathbf{X}_{t} = e^{t}\mathbf{X}_{t}\mathrm{d}t + e^{t}(-\mathbf{X}_{t}\mathrm{d}t + \mathrm{d}\mathbf{B}_{t}) = e^{t}\mathrm{d}\mathbf{B}_{t}$. 
Then, $\mathbf{X}_{t}= e^{-t}(\int_{0}^{t}e^{s}\mathrm{d}\mathbf{B}_{s} + \mathbf{X}_{0})$
Finally, $\mathbf{X}_{t} \sim \mathcal{N}\left(e^{-t}\mathbf{X}_{0}, {\color{red}{(e^{-2t}\int_{0}^{t}e^{2s}\mathrm{d}s) \mathbb{I}_{d}}}\right) = \mathcal{N}\left(e^{-t}\mathbf{X}_{0}, {\color{red}{\frac{1-e^{-2t}}{2} \mathbb{I}_{d}}}\right)$

**e.g.** **Langevin diffusion**. $\mathrm{d}\mathbf{X}_{t}=- \frac{1}{2}\nabla f(\mathbf{X}_{t})\mathrm{d}t + \mathrm{d}\mathbf{B}_{t}$. Convergence: Next lecture

**e.g.** **Black-Scholes model**. $\mathrm{d}X_{t} = \mu X_{t}\mathrm{d}t + \sigma X_{t} \mathrm{d}B_{t}$, $X_{0}\gt 0$
Let $S_{t}=X_{0}\exp(at + b B_{t})$, calculate that $\mathrm{d} S_{t} = aX_{0}\exp(at + bB_{t})\mathrm{d}t + bX_{0}\exp(at+bB_{t})\mathrm{d}B_{t} + \frac{1}{2}b^{2}X_{0}\exp(at+bB_{t})\mathrm{d}t$
Then, $\mathrm{d}S_{t} = (a+\frac{b^{2}}{2})X_{0}\exp(at + bB_{t})\mathrm{d}t + bX_{0}\exp(at+bB_{t})\mathrm{d}B_{t} = \left(a+\frac{b^{2}}{2}\right)S_{t}\mathrm{d}t + bS_{t}\mathrm{d}B_{t}$
Let $a=\mu - \frac{\sigma^{2}}{2}$, $b = \sigma$, $X_{t} = X_{0}\exp((\mu - \frac{\sigma^{2}}{2})t + \sigma B_{t})$ is the solution. "geometric BM"
- e.g. If $\mu \lt \frac{\sigma^{2}}{2}$, $t \to 0$, $X_{t} \to 0$ (exponentially fast). In ML, this corresponds to a stochastic optimization problem.

# Lecture 2. Fokker-Planck equation, diffusion processes, convergence of diffusion process, coupling and mixing time

## Diffusion process
For Diffusion process(time- homogeneous): $\mathrm{d}\mathbf{X}_{t} = \mathbf{b}(\mathbf{X}_{t})\mathrm{d}t + \mathbf{\Sigma}(\mathbf{X}_{t})^{\frac{1}{2}}\mathrm{d}\mathbf{B}_{t}$
We want to formulate its characterization:
**e.g.** Analogy to Kolmogorov-Chapman Equation in Markov Chain
We have $X_{0}$, $\cdots$, $X_{t}$ in Markov Chain with transition matrix $\mathbf{P}$
- If $X_{0} \sim \pi_{0}$, then $X_{t} \sim \pi_{0}\mathbf{P}^{t}$
- For function $f$ on state space, $\mathbb{E}[f(X_{t})\mid X_{0}=x] = (\mathbf{P}^{t}f)(x)$
Important properties: **Stationary**, **Convergence**, **Rate**...

**For the first question**
Suppose we have a (**smooth** enough) potential function $f$
Question: What is $\mathbb{E}[f(\mathbf{X}_{t})\mid \mathbf{X}_{0}=\mathbf{x}]$ where $\mathbf{X}_{t}$ follows $\mathrm{d}\mathbf{X}_{t} = \mathbf{b}(\mathbf{X}_{t})\mathrm{d}t + \mathbf{\Sigma}(\mathbf{X}_{t})^{\frac{1}{2}}\mathrm{d}\mathbf{B}_{t}$
$$\begin{align}
\mathbb{E}[f(\mathbf{X}_{T})\mid \mathbf{X}_{0}=\mathbf{x}] &= f(\mathbf{x}) + 
\mathbb{E}\bigg[\int_{0}^{T}\nabla f(\mathbf{X}_{t})^{\top}\mathbf{b}(\mathbf{X}_{t})\mathrm{d}t + \underbrace{{\color{red}{\cancel{\int_{0}^{T}\nabla f(\mathbf{X}_{t})^{\top}\mathbf{\Sigma}(\mathbf{X}_{t})^{\frac{1}{2}}\mathrm{d}\mathbf{B}_{t}}}}}_{\text{martingale}} \\
&\quad\quad\quad\quad\quad +\frac{1}{2} \int_{0}^{T}\mathrm{Tr}(\nabla^{2}f(\mathbf{X}_{t})\mathbf{\Sigma}(\mathbf{X}_{t}))\mathrm{d}t \bigg] \\
&= f(\mathbf{x}) + \mathbb{E}\left[\int_{0}^{T}\left[\nabla f(\mathbf{X}_{t})^{\top}\mathbf{b}(\mathbf{X}_{t}) + 
\frac{1}{2} \mathrm{Tr}(\nabla^{2}f(\mathbf{X}_{t})\mathbf{\Sigma}(\mathbf{X}_{t})) \right]\mathrm{d}t\right]
\end{align}$$
Then, calculate that
$$\lim\limits_{T \to 0^{+}} \frac{\mathbb{E}[f(\mathbf{X}_{T})\mid \mathbf{X}_{0}=\mathbf{x}] - f(\mathbf{x})}{T}
= \nabla f(\mathbf{X}_{t})^{\top}\mathbf{b}(\mathbf{X}_{t}) + 
\frac{1}{2} \mathrm{Tr}(\nabla^{2}f(\mathbf{X}_{t})\mathbf{\Sigma}(\mathbf{X}_{t}))$$
Analogous to Markov Chains, we define a operator $(\mathcal{P}_{t})_{t\ge 0}$ satisfying $(\mathcal{P}_{t}f)(\mathbf{x}) = \mathbb{E}[f(\mathbf{X}_{t})\mid \mathbf{X}_{0}=\mathbf{x}]$. We call $(\mathcal{P}_{t})_{t\ge 0}$ as semi-group
- Identity element: $\mathcal{P}_{0} = \mathbb{I}: f \mapsto f$
- Multiplication: $\mathcal{P}_{t} \circ \mathcal{P}_{s} = \mathcal{P}_{t+s}$
- No inverse element.
Let $\mathcal{A} = \frac{\mathrm{d}\mathcal{P}_t}{\mathrm{d}t} \bigg|_{t=0}$, then $\mathcal{A}f(\mathbf{x}) = \nabla f(\mathbf{X}_{t})^{\top}\mathbf{b}(\mathbf{X}_{t}) + \frac{1}{2} \mathrm{Tr}(\nabla^{2}f(\mathbf{X}_{t})\mathbf{\Sigma}(\mathbf{X}_{t}))$
$\mathcal{A}$ is a linear operator, $\mathcal{A}(\lambda f_{1}+ \mu f_{2}) = \lambda \mathcal{A}f_{1} + \mu \mathcal{A}f_{2}$
**e.g.** For finite-dim matrix $\mathbf{A}$, if $\frac{\mathrm{d}{P}_t}{\mathrm{d}t} = \mathbf{A} {P}_t$, then $P_{t}=P_{0}\exp(t \mathbf{A})$

## Fokker-Planck Equation
Let $\pi_{t}$ be the density of $\mathbf{X}_{t}$, then,
$$\begin{align}
\frac{\mathrm{d}}{\mathrm{d}t} \mathbb{E}\left[f(\mathbf{X}_{t})\right] 
&= \frac{\mathrm{d}}{\mathrm{d}t}\int_{\mathbb{R}^{d}} f(\mathbf{x}) \pi_{t}(\mathbf{x})\mathrm{d}\mathbf{x}
\\&= \int_{\mathbb{R}^{d}} f(\mathbf{x}) \frac{\partial\pi_{t}}{\partial t}(\mathbf{x}) \mathrm{d}\mathbf{x}
\end{align}$$
On the other hand,
$$\begin{align}
\frac{\mathrm{d}}{\mathrm{d}t} \mathbb{E}\left[f(\mathbf{X}_{t})\right] 
&= \frac{\mathrm{d}}{\mathrm{d}t} \mathbb{E}\left[(\mathcal{P}_{t}f)(\mathbf{X}_{0})\right] 
\\&= \mathbb{E}\left[\frac{\mathrm{d}}{\mathrm{d}t} (\mathcal{P}_tf)(\mathbf{X}_{0})\right]
\\&=\mathbb{E}\left[(\mathcal{A}\mathcal{P}_{t}f)(\mathbf{X}_{0}) \right]
\end{align}$$
Let $t=0$, then,
$$\int_{\mathbb{R}^{d}}  f(\mathbf{x}) \frac{\partial\pi_{t}(\mathbf{x})}{\partial t}\bigg|_{t=0} \mathrm{d}\mathbf{x} = \int_{\mathbb{R}^{d}} \mathcal{A}f(\mathbf{x})\pi_{0}(\mathbf{x})\mathrm{d}\mathbf{x}$$
>[!note] Integration-by-parts formulae
>Single variable: $$\color{red}\boxed{\int_{a}^{b}f^{\prime}(x)g(x) \mathrm{d}x = f(x)g(x)\bigg|_{a}^{b} - \int_{a}^{b}f(x)g^{\prime}(x)\mathrm{d}x}$$
>Multivariate: For bounded domain $\Omega$ (with smooth boundary), if $f(x)g(x)\bigg|_{\partial\Omega}=0$, then
>$$\color{red}\boxed{\int_{\Omega} \nabla f(x) g(x) \mathrm{d}x = -\int_{\Omega} f(x) \nabla g(x) \mathrm{d}x}$$
In general, we have
$$\color{red}\boxed{\int_{\Omega} \nabla f(x)g(x) \mathrm{d}x = \oint_{\partial \Omega} f(x)g(x) \mathrm{d}\mathbf{n}(x) -\int_{\Omega} f(x)\nabla g(x) \mathrm{d}x}$$

A non-rigorous proof: Let $\Omega = \mathbb{B}(0, \mathbb{R})$, then
$$\begin{align}
\left|
\int_{\Omega}f(\mathbf{x}) \partial_{t} \pi_{0}\mathrm{d}\mathbf{x} - 
\int_{\mathbb{R}^d}f(\mathbf{x}) \partial_{t} \pi_{0}\mathrm{d}\mathbf{x}
\right| \to 0
\end{align}$$
Similarly for $\mathcal{A}f \cdot \pi_{0}$.
$$\begin{align}
\int_{\Omega} \mathcal{A}f(\mathbf{x}) \pi_{0}(\mathbf{x}) \mathrm{d}\mathbf{x} 
&= \int_{\Omega}\nabla f(\mathbf{x})^{\top}\mathbf{b}(\mathbf{x}) \pi_{0}(\mathbf{x}) \mathrm{d}\mathbf{x} + \frac{1}{2} \int_{\Omega}\mathrm{Tr}(\mathbf{\Sigma}(\mathbf{x}) \nabla^{2}f(\mathbf{x}))\pi_{0}(\mathbf{x})\mathrm{d}\mathbf{x}
\\&=\int_{\Omega} \left[ \nabla f(\mathbf{x})^{\top}\mathbf{b}(\mathbf{x}) \pi_{0}(\mathbf{x}) \mathrm{d}\mathbf{x} + \frac{1}{2} \mathrm{Tr}(\mathbf{\Sigma}(\mathbf{x}) \nabla^{2}f(\mathbf{x}))\pi_{0}(\mathbf{x}) \right] \mathrm{d}\mathbf{x}
\end{align}$$
**For the first term**,
$$\begin{align}
\text{First term} &=
\oint_{\partial \Omega}\pi_{0}(\mathbf{x}) f(\mathbf{x}) \mathbf{b}(\mathbf{x})^{\top}\mathrm{d}\mathbf{n}(\mathbf{x}) - \int_{\Omega}f(\mathbf{x}) \nabla(\pi_{0}\mathbf{b})(\mathbf{x}) \mathrm{d}\mathbf{x}
\end{align}$$
as $\Vert \mathbf{x} \Vert \to + \infty$, $f(\mathbf{x}) \mathbf{b}(\mathbf{x})^{\top}$ grows polynomially; while in $\left|\partial \Omega\right| = C_{d}\mathbb{R}^d$, $\pi_{0}(\mathbf{x}) \le C_{1}\exp(-C_{2}\Vert \mathbf{x} \Vert^{2})$.
Then, $\oint_{\partial \Omega}\pi_{0}(\mathbf{x}) f(\mathbf{x}) \mathbf{b}(\mathbf{x})^{\top}\mathrm{d}\mathbf{n}(\mathbf{x}) \to 0$ at $\partial \Omega$.

**For the second term**,
$$\begin{align}
\text{Second term} &=  \int_{\mathbb{R}^{d}}\mathrm{Tr}(\nabla^{2}f(\mathbf{x})\mathbf{\Sigma}(\mathbf{x}))\pi_{0}(\mathbf{x})\mathrm{d}\mathbf{x}
\\&=-\int_{\mathbb{R}^{d}} \nabla f(\mathbf{x}) \cdot (\nabla \cdot \pi \mathbf{\Sigma})(\mathbf{x}) \mathrm{d}\mathbf{x}
\\&=\int_{\mathbb{R}^{d}}f(\mathbf{x}) \nabla^{2}(\pi \mathbf{\Sigma})(\mathbf{x})\mathrm{d}\mathbf{x}
\end{align}$$
(For simplicity, ignore $\text{boundary term} \to 0$)

**Putting them together**,
$$\begin{align}
\int f(\mathbf{x}) \partial_{t}\pi_{0}(\mathbf{x})\mathrm{d}\mathbf{x} = \int f(\mathbf{x}) \left[-\nabla \cdot (\pi_{0}\mathbf{b})(\mathbf{x}) + \frac{1}{2} \nabla^{2}(\pi_{0}\mathbf{\Sigma})(\mathbf{x}) \right] \mathrm{d}\mathbf{x}
\end{align}$$
so (by time-homogenous), we get **Fokker-Planck equation**:
$$\color{red}\boxed{\partial_{t}\pi_{t}(x) = -\nabla \cdot (\pi_{t}\mathbf{b})(\mathbf{x}) + \frac{1}{2} \nabla^{2}(\pi_{t}\mathbf{\Sigma})(\mathbf{x})}$$
**Remark**: we got adjoint operator: $\mathcal{A}^{\star}\pi = -\nabla (\pi_{t}\mathbf{b})(\mathbf{x}) + \frac{1}{2} \nabla^{2}(\pi_{t}\mathbf{\Sigma})(\mathbf{x})$
Let $\langle f, g\rangle = \int_{\mathbb{R}^{d}} fg \mathrm{d}\mathbf{x}$, then we have $\langle \mathcal{A}f, \pi_{t} \rangle = \langle f, \partial_{t} \pi \rangle$, so $\mathcal{A}^{\star}\pi = \partial_{t} \pi = -\nabla \cdot (\pi_{t}\mathbf{b})(\mathbf{x}) + \frac{1}{2} \nabla^{2}(\pi_{t}\mathbf{\Sigma})(\mathbf{x})$

**e.g.** $\mathrm{d}\mathbf{X}_{t} = \mathrm{d}\mathbf{B}_{t}$. Then, we get $\partial_{t}\pi_{t} = \frac{1}{2} \Delta \pi_{t}$. (Fundamental solution: Gaussian convolution)

**e.g.(Langevin)**. $\mathrm{d}\mathbf{X}_{t} = -\nabla f(\mathbf{X}_{t})\mathrm{d}t + \sqrt{2} \mathrm{d}\mathbf{B}_{t}$. Then, we get $\partial_{t}\pi_{t} = \nabla \cdot (\pi \nabla f) + \Delta \pi_{t}$

## Sampling Problem and Coupling
Given $f: \mathbb{R}^{d} \to \mathbb{R}$, generate a sample from $\pi(\mathbf{x}) = \frac{\exp(-f(\mathbf{x}))}{\int_{\mathbb{R}^{d}} \exp(-f(\mathbf{y}))\mathrm{d}\mathbf{y}}$
Basic idea:
- Importance sampling
- Rejection sampling
scales poorly with high dim.

**MCMC** (Markov Chain Monte Carlo): Run a stochastic process with **stationary distribution** $\pi$

**Stationary distribution of Langevin diffusion**
Recall, $\partial_{t}\pi_{t} = \nabla \cdot (\pi \nabla f) + \Delta \pi_{t}$. Its stationary condition is $\nabla \cdot (\pi \nabla f) + \Delta \pi = 0$
Let $\pi = \exp(-f)$, then, 
$$\begin{align}
\nabla \cdot (\pi \nabla f) &= \nabla \cdot(\exp(-f) \nabla f)
\\&=\Delta f \cdot \exp(-f) - |\nabla f|^{2}\exp(-f)
\end{align}$$
$$\begin{align}
\Delta exp(-f) &= \nabla \cdot (\nabla \exp(-f)) \\ 
&=\nabla \cdot (-\nabla f \cdot \exp(-f)) \\
&=-\Delta f \cdot \exp(-f) + |\nabla f|^{2}\exp(-f)
\end{align}$$
so $\pi \propto \exp(-f)$ is a stationary distribution.
- $\exists !$ under weak condition.
- Convergence (rate) in what sense?

**Distance** (divergence) between probability distribution:
**Total variation**:
$$d_{\mathrm{TV}}(\mathcal{P}, \mathcal{Q}) = \frac{1}{2}\int\left|p(x) - q(x)\right|\mathrm{d}x = \sup\limits_{\Vert f\Vert_{\infty} \le 1} \left|\mathbb{E}_\mathcal{P}f(x) - \mathbb{E}_\mathcal{Q}f(x)\right|$$
**Kullback-Leibler**:
$$\mathcal{D}_\mathrm{KL}(\mathcal{P}\Vert \mathcal{Q}) = \mathbb{E}_\mathcal{P} \left[\log \frac{p(x)}{q(x)}\right]$$
**Pinsker's inequality**:
$$d_\mathrm{TV}(\mathcal{P}, \mathcal{Q}) \le \sqrt{\frac{1}{2} \mathcal{D}_\mathrm{KL}(\mathcal{P}\Vert \mathcal{Q})}$$

**Wasserstein distance**
Let $X \sim \mathcal{P}$, $Y \sim \mathcal{Q}$.
**Coupling**: Find a joint distribution $\gamma$, s.t. $\gamma\bigg|_{X}= \mathcal{P}$, $\gamma\bigg|_{Y} = \mathcal{Q}$.
Then we can define **Wasserstein distance** as follows,
$$\begin{align}
\mathcal{W}_{1}(\mathcal{P}, \mathcal{Q}) &= {
\inf\limits_{\gamma \in \mathrm{Coupling}(\mathcal{P}, \mathcal{Q})} \mathbb{E}_\gamma\left[\Vert X - Y \Vert\right]
}
\\&=\sup\limits_{\Vert\nabla f \Vert_{2} \le 1} \left|\mathbb{E}_\mathcal{P} f(x) - \mathbb{E}_\mathcal{Q} f(x) \right|
\\
\mathcal{W}_{2}(\mathcal{P}, \mathcal{Q}) &= \sqrt{
\inf\limits_{\gamma \in \mathrm{Coupling}(\mathcal{P}, \mathcal{Q})} \mathbb{E}_\gamma\left[\Vert X - Y \Vert^{2}\right]
}
\end{align}$$
By Cauchy-Schwarz inequality, $\mathcal{W}_{1}(\mathcal{P}, \mathcal{Q}) \le \mathcal{W}_{2}(\mathcal{P}, \mathcal{Q})$

### Convergence rate of Langevin diffusion
Assumption: $\nabla^{2}f \succeq \lambda \mathbb{I}_{d}$, $\forall \mathbf{x}$, for some $\lambda \gt 0$
Strong convexity (assumption above) implies
$$\langle \nabla f(\mathbf{x}) - \nabla f(\mathbf{x}), \mathbf{x} - \mathbf{y} \rangle \ge \lambda \Vert \mathbf{x} - \mathbf{y} \Vert^{2}$$
>[!proof]
>By the following calculation, it is trivial.
>$$\nabla f(\mathbf{x}) - \nabla f(\mathbf{y}) = \int_{0}^{1}\nabla^{2}f\left(t \mathbf{x} + (1-t) \mathbf{y}\right)(\mathbf{x}-\mathbf{y})\mathrm{d}t$$

**Goal**: construct a coupling between $\mathbf{X}_{T}$ (time $T$ marginal ($\pi_{T}$) of diffusion) and $\mathbf{X}^{\star}$(distribution $\pi$)
Let $\mathrm{d}\mathbf{X}_{t} = -\nabla f(\mathbf{X}_{t})\mathrm{d}t + \sqrt{2}\mathrm{d}\mathbf{B}_{t}$, $\mathbf{X}_{0} = \mathbf{x}_{0}$ and $\mathrm{d}\mathbf{X}_{t}^{\star} = -\nabla f(\mathbf{X}_{t}^{\star})\mathrm{d}t + \sqrt{2}\mathrm{d}\mathbf{B}_{t}$, $\mathbf{X}_{0}^{\star} \sim \pi$.
We have $\mathbf{X}_{T}^{\star}\overset{\mathrm{d}}{=}\mathbf{X}^{\star}$. We want to construct "Synchronous coupling": same BM.
Let $\mathcal{H}_{t} \coloneqq \mathbb{E}[\Vert \mathbf{X}_{t} - \mathbf{X}_{t}^{\star}\Vert^{2}]$
Calculate $\mathrm{d}(\mathbf{X}_{t} - \mathbf{X}_{t}^{\star}) = -(\nabla f(\mathbf{X}_{t}) - \nabla f(\mathbf{X}_{t}^{\star}))\mathrm{d}t$
Then, 
$$\begin{align}
\frac{\mathrm{d}\mathcal{H}_{t}}{\mathrm{d}t} &=
\frac{\mathrm{d}}{\mathrm{d}t}\mathbb{E}[\Vert \mathbf{X}_{t} - \mathbf{X}_{t}^{\star}\Vert^{2}]
\\&=-2 \mathbb{E}\left[\langle \mathbf{X}_{t} - \mathbf{X}_{t}^{\star}, \nabla f(\mathbf{X}_{t}) - \nabla f(\mathbf{X}_{t}^\star) \rangle\right]
\\&\le-2 \lambda \mathbb{E}\left[ \Vert \mathbf{X}_{t} - \mathbf{X}_{t}^{\star}\Vert^{2} \right]
\\&=-2 \lambda \mathcal{H}_{t}
\end{align}$$
Let $\Phi_{t} = \exp(2 \lambda t) \mathcal{H}_{t}$, then $\frac{\mathrm{d}\Phi}{\mathrm{d}t} \le 0$, so $\Phi_{t} \le \Phi_{0}=\mathcal{H}_{0}$, so $\mathcal{H}_{t} \le \exp(-2\lambda t)\mathcal{H}_{0}$.
Then,
$$\mathcal{W}_{2}^{2}(\pi_{T}, \pi) = \inf \mathcal{H}_{T}\le \mathcal{H}_{T} \le \exp(-2\lambda T) \mathbb{E}\left[\Vert \mathbf{X}_{0} - \mathbf{X}_{0}^{\star}\Vert^{2} \right] \to 0$$
**Extension**. Underdamped Langevin
[[1707.03663] Underdamped Langevin MCMC: A non-asymptotic analysis](https://arxiv.org/abs/1707.03663)
$$\begin{cases}
\mathrm{d}\mathbf{v}_{t} = -\gamma \mathbf{v}_{t}\mathrm{d}t - u \nabla f(\mathbf{x}_{t})\mathrm{d}t + \sqrt{2\gamma u}\mathrm{d}\mathbf{B}_{t} \\
\mathrm{d}\mathbf{x}_{t} = \mathbf{v}_{t} \mathrm{d}t
\end{cases}$$
Use Fokker-Planck equation, let $\pi_{t}: \mathbb{R}^{2d} \to \mathbb{R}_{+}$ be the density function.
Then, 
$$\boxed{\partial_{t}\pi_{t} = \gamma \nabla_{\mathbf{v}}\cdot(\pi \mathbf{v}) + u \nabla_{\mathbf{v}}\cdot(\pi \nabla f(\mathbf{x})) - \nabla_{\mathbf{x}}\cdot(\pi \mathbf{v}) + \gamma u \Delta_{\mathbf{v}}\pi}$$
It stationary distribution: $\pi(\mathbf{x}, \mathbf{v}) \propto exp(-(f(\mathbf{x})+ \frac{ \Vert \mathbf{v}\Vert_{2}^{2} }{2u}))$

**Convergence**: still synchronous coupling (assuming strong convexity)
Construct Lyapunov function:
$$\begin{align}
\mathcal{H}_{t} &= \mathbb{E}\left[\left\Vert 
\begin{pmatrix} 1  & 1 \\ 0 & 1\end{pmatrix}
\begin{pmatrix} \mathbf{x}_{t} - \mathbf{x}_{t}^{\star} \\ \mathbf{v}_{t} - \mathbf{v}_{t}^{\star}\end{pmatrix}
\right\Vert_{2}^{2}\right]
\end{align}$$
Let$\mathbf{z}_{t} = \mathbf{x}_{t} - \mathbf{x}_{t}^{\star}$, $\mathbf{y}_{t}=\mathbf{v}_{t} - \mathbf{v}_{t}^{\star}$, then, 
$$\begin{align}
\frac{\mathrm{d}\mathcal{H}_{t}}{\mathrm{d}t} &= 
2\mathbb{E}\left[
\begin{pmatrix}
\mathbf{z}_{t} + \mathbf{y}_{t} &  \mathbf{y}_{t} 
\end{pmatrix}
\begin{pmatrix}
(1-\gamma)\mathbf{y}_{t} - u \left(\nabla f(\mathbf{x}_{t}) - \nabla f(\mathbf{x}_{t}^{\star})\right) \\ -\gamma\mathbf{y}_{t}  - u \left(\nabla f(\mathbf{x}_{t}) - \nabla f(\mathbf{x}_{t}^{\star})\right)
\end{pmatrix}
\right]
\end{align}$$
>[!note] Detour
>Stability of dynamic system $\mathrm{d}\mathbf{x}_{t} = \mathbf{A}\mathbf{x}_{t}\mathrm{d}t$
>If $\frac{\mathbf{A}+\mathbf{A}^{\top}}{2}$ has negative eigenvalues, then exp. converge to 0

$\nabla f$ is nonlinear, but we state that $\nabla f(\mathbf{x}_{t}) - \nabla f(\mathbf{x}_{t}^{\star}) = \int_{0}^{1} \nabla^{2}f(\beta \mathbf{x}_{t} + (1-\beta)\mathbf{x}_{t}^{\star})( \mathbf{x}_{t} - \mathbf{x}_{t}^{\star} )\mathrm{d}\beta$
Assume: $\lambda \mathbb{I} \preceq \nabla^{2} f \preceq L \mathbb{I}$
So, $\mathbf{\Lambda}_{t} = \int_{0}^{1} \nabla^{2}f(\beta \mathbf{x}_{t} + (1-\beta)\mathbf{x}_{t}^{\star})\mathrm{d}\beta \in \left[\lambda \mathbb{I}, L \mathbb{I}\right]$
Then, 
$$\begin{align}
\frac{\mathrm{d}\mathcal{H}_{t}}{\mathrm{d}t} &= -
\begin{pmatrix} \mathbf{x}_{t} - \mathbf{x}_{t}^{\star} & \mathbf{v}_{t} - \mathbf{v}_{t}^{\star}\end{pmatrix}
\begin{pmatrix} (\gamma-1)\mathbb{I}_{d} & u \mathbf{\Lambda}_{t}-(\gamma-1)\mathbb{I}_{d} \\ -\mathbb{I}_{d} & \mathbb{I}_d \end{pmatrix}
\begin{pmatrix} \mathbf{x}_{t} - \mathbf{x}_{t}^{\star} \\ \mathbf{v}_{t} - \mathbf{v}_{t}^{\star}\end{pmatrix}
\\&=\begin{pmatrix}
\mathbf{z}_{t} + \mathbf{y}_{t}  &  \mathbf{y}_{t} 
\end{pmatrix}
\mathbf{\delta}_{t}
\begin{pmatrix}
\mathbf{z}_{t} + \mathbf{y}_{t}\\ \mathbf{y}_{t} 
\end{pmatrix}
\end{align}$$
Need $\frac{\mathbf{\delta}_{t}+\mathbf{\delta}_{t}^{\top}}{2}$ negative definite, take $\gamma=2$, $u=\frac{1}{L} \Rightarrow \lambda_{\mathrm{min}}(-\frac{\mathbf{\delta}_{t}+\mathbf{\delta}_{t}^{\top}}{2})\ge \frac{\lambda}{2L}$

# Lecture 3. Functional inequalities and diffusion process convergence

For DTMC (Discrete-time Markov Chain): "Mixing time"
- Coupling
- Spectral gap
**Spectral gap**:
Let stationary state $\pi = \pi \mathbf{P}$, where $\pi \in \mathbb{R}^{l\times n}$, $\mathbf{P} \in \mathbf{R}^{n\times n}$. We have $\pi_{n} = \pi_{0}\mathbf{P}^{n}$. Diagonalize $\mathbf{P} = \mathbf{D} \mathbf{\Lambda} \mathbf{D}^{-1}$, then $\pi_{n} = \pi_{0}\mathbf{D} \mathbf{\Lambda}^{n} \mathbf{D}^{-1}$. Its convergence rate depends on $\frac{1}{1-|\lambda_{2}|}$

## Functional Inequalities
"The correct analogue of spectral gap"
### log-Sobolev Inequality

**Motivation**.
$$\begin{align}
\frac{\mathrm{d}}{\mathrm{d}t} \mathcal{D}_\mathrm{KL} (\pi_{t}\Vert \pi)
&= \int\partial_{t}(\pi_{t} \log \frac{\pi_{t}}{\pi})\mathrm{d}\mathbf{x}
\\&= \int\partial_{t}\pi_{t} \log \frac{\pi_{t}}{\pi} \mathrm{d}\mathbf{x}
+ \int \pi_{t} \partial_{t}\log \pi_{t} \mathrm{d}\mathbf{x}
\\&=\int\left(\nabla \cdot (\pi_{t} \nabla f) + \Delta \pi_{t} \right)\left(\log \frac{\pi_{t}}{\pi} + 1\right)\mathrm{d}\mathbf{x}
\\&=-\int(\pi_{t}\nabla f + \nabla \pi_{t})^{\top}\left(\nabla \log \frac{\pi_{t}}{\pi}\right) \mathrm{d}\mathbf{x} \quad\quad\left( f = -\log\pi, \nabla\pi_{t}=\pi_{t}\nabla\log\pi_{t} \right)
\\&=-\int\Vert \nabla\log\pi_{t}-\nabla\log\pi \Vert_{2}^{2}\pi_{t}\mathrm{d}\mathbf{x}
\\&=-\int\pi_{t} \Vert \nabla\log \frac{\pi_{t}}{\pi} \Vert_{2}^{2}\mathrm{d}\mathbf{x}
\end{align}$$
**Def**. We say $\pi$ satisfies log-Sobolev inequality with constant $C_\mathrm{LSI}$ iff for $\forall \mu$ (sufficiently smooth),
$$
\mathcal{D}_\mathrm{KL}(\mu \Vert \pi) \le C_\mathrm{LSI} \underbrace{{\color{red}{\int \mu(\mathbf{x})\Vert \nabla\log \frac{\mu}{\pi} \Vert_{2}^{2} \mathrm{d}\mathbf{x}}}}_{I(\mu\Vert\pi)\text{: Relative Fisher Information}}
$$
Suppose that LSI holds, we have $$\color{red}{\boxed{\mathcal{D}_\mathrm{KL}(\pi_{t}\Vert\pi) \le\mathcal{D}_\mathrm{KL}(\pi_{0}\Vert\pi)\exp(-\frac{t}{C_\mathrm{LSI}})}}$$**Proof of LSI under strong convexity condition**. (This is an instance of "**interpolation method**")
$$\begin{align}
\frac{\mathrm{d}}{\mathrm{d}t}I(\pi_{t}\Vert\pi)
&= \underbrace{-2 \mathbb{E}\left[\Vert \nabla^{2}\log \frac{\pi_{t}}{\pi} \Vert_{F}^{2}\right]}_{\le 0}

\underbrace{-2\mathbb{E}\left[(\nabla \log \frac{\pi_{t}}{\pi})^{\top}\nabla^{2}f(\nabla\log \frac{\pi_{t}}{\pi}) \right]}_{\le-2 \lambda I(\pi_{t}\Vert\pi)}

\\&\le -2\lambda I(\pi_{t}\Vert\pi)\quad\quad\quad\text{(Under strong convexity)}

\\ &\text{(Then $I(\pi_{t}\Vert\pi) \le I(\pi_{0}\Vert\pi) \exp(-2\lambda t)$)}

\end{align}$$
Also, we have
$$\begin{align}
\frac{\mathrm{d}}{\mathrm{d}t}\mathcal{D}_\mathrm{KL}(\pi_{t}\Vert\pi) = -I(\pi_{t}\Vert\pi)
\end{align}$$
So we have
$$\begin{align}
\mathcal{D}_\mathrm{KL}(\pi_{0}\Vert\pi) &=
\int_{0}^{\infty}I(\pi_{t}\Vert\pi)\mathrm{d}t
\\&\le \int_{0}^{\infty}I(\pi_{0}\Vert\pi) \exp(-2\lambda t)\mathrm{d}t
\\&=\frac{1}{2\lambda}I(\pi_{0}\Vert\pi)
\end{align}$$

### Poincare Inequality

**Motivation**. Let 

$$\begin{align}
\chi^{2}(\mathcal{P}\Vert\mathcal{Q}) &= 
\int q(\mathbf{x})\left(\frac{p(\mathbf{x})}{q(\mathbf{x})}-1\right)^{2}\mathrm{d}\mathbf{x}
\\&=\int p(\mathbf{x})\left(\frac{p(\mathbf{x})}{q(\mathbf{x})}-1\right) \mathrm{d}\mathbf{x}
\end{align}$$
 Then
$$
\begin{align}
\frac{\mathrm{d}}{\mathrm{d}t} \chi^{2}(\pi_{t}\Vert\pi)
&=\frac{\mathrm{d}}{\mathrm{d}t} \int\pi_{t}\left( \frac{\pi_{t}}{\pi}-1 \right)\mathrm{d}\mathbf{x}
\\&= \int \left(\partial_{t}\pi_{t}(\frac{\pi_{t}}{\pi}-1) + \pi_{t}\frac{\partial_{t}\pi_{t}}{\pi}\right)\mathrm{d}\mathbf{x}
\\&= \int \partial_{t}\pi_{t} \left(\frac{2\pi_{t}}{\pi} - 1\right)\mathrm{d}\mathbf{x}
\\&= \int\left(\nabla \cdot (\pi_{t} \nabla f) + \Delta \pi_{t} \right) \left( \frac{2\pi_{t}}{\pi} - 1\right) \mathrm{d}\mathbf{x}
\\&=-\int (\pi_{t} \nabla f + \nabla\pi_{t}) \frac{2\nabla\pi_{t}}{\pi} \mathrm{d}\mathbf{x}  \quad\quad\left( f = -\log\pi, \nabla\pi_{t}=\pi_{t}\nabla\log\pi_{t} \right)
\\&=-2\int \pi_{t}(\nabla\log \frac{\pi_{t}}{\pi})\nabla\left(\frac{\pi_{t}}{\pi}\right) \mathrm{d}\mathbf{x}
\quad\left(\nabla\frac{\pi_{t}}{\pi}= \frac{\pi_{t}}{\pi} \nabla\log \frac{\pi_{t}}{\pi} \right)
\\&=-2\int\pi \Vert \nabla\left(\frac{\pi_{t}}{\pi}\right)  \Vert_{2}^{2}\mathrm{d}\mathbf{x}
\end{align}
$$
**Def**. We say $\pi$ satisfies Poincare Inequality with constant $C_\mathrm{PI}$ iff for $\forall h$ (sufficiently smooth function),
$$
\mathrm{Var}_\pi(h(\mathrm{x})) \le C_\mathrm{PI} \mathbb{E}\left[\Vert\nabla h \Vert^{2}\right]
$$
Suppose that PI holds, we have 
$$\color{red}{\boxed{\chi^{2}(\pi_{t}\Vert\pi)\le\chi^{2}(\pi_{0}\Vert\pi)\exp(- \frac{t}{C_{PI}})}}$$
**Intuition**. Functional inequality $\approx$ Isoperimetry Inequality $\approx$ Difficult to find a cut
Cut $S=S_{1}\oplus S_{2}$, a "good" cut is that: $\mathbb{P}(S_{1})$ and $\mathbb{P}(S_{2})$ big, while $\mathbb{P}(\partial S_{1} \setminus \partial S)$ small

**Isoperimetry.** $$\min\limits_{S_{1}} \frac{\mathbb{P}(\partial S_{1} \setminus \partial S)}{\min(\mathbb{P}(S_{1}), \mathbb{P}(S_{2}))}$$
<div align="center">
<img src="figures/Continuous-time Diffusion/isoperimetry.png" width="450">
</div>

Suppose we have a good cut, we can construct a continuous function valued $0$ in area $S_{1}$, valued $1$ in area $S_{2}$, and interpolation from $0$ to $1$ near $\partial S_{1}$. Then we have $\mathbb{E}\left[\Vert\nabla f \Vert^{2}\right]$ small, while $\mathrm{Var}(f)$ large.

### log-Sobolev inequality and Poincare inequality

>[!note] Compare:
>PI: $\mathrm{Var}_{\pi}(f)\le C_\mathrm{PI}\mathbb{E}_{\pi}[\Vert\nabla f\Vert^{2}]$
>LSI: $\mathcal{D}_\mathrm{KL}(\mu\Vert\pi) \le C_\mathrm{LSI} I(\mu\Vert\pi)$

**Beyond convexity**: An equivalent form of log-Sobolev inequality: 
Change of variable: Let $f = C_{\mathrm{LSI}}\sqrt{ \frac{\mu}{\pi} }$
LSI is equivalent to (for sufficiently smooth function $f$)
$$\color{red}{\boxed{\mathbb{E}_{\pi}[f^{2}\log f^{2}] - \mathbb{E}_{\pi}[f^{2}]\mathbb{E}_{\pi}[\log f^{2}] \le 4C_\mathrm{LSI}\mathbb{E}_{\pi}[\Vert\nabla f\Vert^{2}]}}$$
**LSI implies PI**:
Apply equivalent form of LSI with $f = 1+\epsilon h$, for small $\epsilon$
$$\begin{align}
\mathrm{LHS} &=
\mathbb{E}_{\pi}\left[ (1+2\epsilon h + \epsilon^{2}h^{2})\cdot
2\left(\epsilon h - \frac{1}{2}\epsilon^{2}h^{2} + O(\epsilon^{3})\right) \right] 
\\&- \mathbb{E}_{\pi}[(1+2\epsilon h + \epsilon^{2}h^{2})]\cdot\mathbb{E}_{\pi}[\log\left(1+2\epsilon h + \epsilon^{2}h^{2}\right)] 

\\&=[2\epsilon\mathbb{E}_{\pi}[h]+3\epsilon^{2}\mathbb{E}_{\pi}[h^{2}] + O(\epsilon^{3})] 
\\&-[1+2\epsilon\mathbb{E}_{\pi}[h] + \epsilon^{2}\mathbb{E}[h^{2}]]\cdot[2\epsilon\mathbb{E}_{\pi}[h]-\epsilon^{2}\mathbb{E}_{\pi}[h^{2}] + O(\epsilon^{3})] 

\\&= 4\epsilon^{2}\mathbb{E}_{\pi}[h^{2}] - 4\epsilon^{2} (\mathbb{E}_{\pi}[h])^{2} + O(\epsilon^{3})
\\&= 4\epsilon^{2}\mathrm{Var}_{\pi}(f) + O(\epsilon^{3})

\\ \mathrm{RHS} &= 4\epsilon^{2}C_\mathrm{LSI}\mathbb{E}_{\pi}[\Vert\nabla h\Vert^{2}]
\end{align}$$
We get
$$4\epsilon^{2}\mathrm{Var}_{\pi}(f)+ O(\epsilon^{3}) \le 4\epsilon^{2}C_\mathrm{LSI} \mathbb{E}_{\pi}[\Vert\nabla h\Vert^{2}]$$
So we get Poincare inequality $\mathbb{E}_{\pi}[h^{2}] \le C_\mathrm{LSI} \mathbb{E}_{\pi}[\Vert\nabla h\Vert^{2}]$, $C_{\mathrm{PI}} \le C_{\mathrm{LSI}}$

### Some distribution and its tail bound

**sub-Gaussian**: 
$$\begin{align}

\mathbb{E}[\exp(\lambda(X-\mathbb{E}X))] &\le \exp(\frac{\lambda^{2}\sigma^{2}}{2})\text{, } \forall \lambda \in \mathbb{R}
\\ \iff \mathbb{P}(|X|\ge t) &\le \exp(- \frac{t^{2}}{C \sigma^{2}})\text{, }\forall t

\end{align}$$
**sub-Gaussian random vector**: $\forall \mathbf{u}$, $\Vert \mathbf{u} \Vert_{2} = 1$, $\mathbf{u}^{\top}\mathbf{X}$ is sub-Gaussian

**sub-exponential**:  
$$\begin{align}

\mathbb{E}[\exp(\lambda(X-\mathbb{E}X))] &\le \exp(\frac{\lambda^{2}\alpha^{2}}{2})\text{, when }\lambda \le \frac{1}{C \alpha} 
\\ \iff \mathbb{P}(|X|\ge t) &\le \exp(- \frac{t}{\alpha})\text{, }\forall t

\end{align}$$

**Poincare inequality implies concentration of Lipschitz function**:
For any $1$-Lipschitz function $g: \mathbb{R}^{d} \to \mathbb{R}$
Let moment generating function $m(\lambda) = \mathbb{E}[\exp(\lambda g(\mathbf{x}))]$
By Poincare inequality,
$$\begin{align}
\mathbb{E}[\exp(2\lambda g(\mathbf{x}))] - (\mathbb{E}[\exp(\lambda g(\mathbf{x}))])^{2} &\le C_{\mathrm{PI}} \lambda^{2} \mathbb{E}[\Vert\nabla g\Vert^{2}\cdot\exp(2\lambda g(\mathbf{x}))] \quad\text{(}g\text{ is 1-Lipschitz)}
\\ &\le \lambda^{2} C_\mathrm{PI} \mathbb{E}[\exp(2\lambda g(\mathbf{x}))]
\end{align}$$
So when $\lambda \lt \frac{1}{\sqrt{2C_\mathrm{PI}}}$,
$$\begin{align}
m(\lambda) &\le \frac{1}{1-\lambda^{2}C_\mathrm{PI}} m^{2}(\frac{\lambda}{2})

\\ &\le \frac{1}{1-\lambda^{2}C_\mathrm{PI}} \left(\frac{1}{1- \frac{\lambda^{2}}{4} C_\mathrm{PI}}\right)^{2}m^{4}(\frac{\lambda}{4})

\\&\le \prod\limits_{k=0}^{n} \left(\frac{1}{1- \frac{1}{4^{k}} \lambda^{2} C_\mathrm{PI}}\right)^{2^{k}} m^{2^{n}} (\frac{\lambda}{2^{n+1}})

\end{align}$$
$$\begin{align}

\log\prod\limits_{k=0}^{n} \left(\frac{1}{1- \frac{1}{4^{k}} \lambda^{2} C_\mathrm{PI}}\right)^{2^k}
&=-\sum\limits_{k=0}^{n}2^{k}\log\left(1-\frac{1}{4^{k}} \lambda^{2}C_\mathrm{PI}\right)

\\&=\sum\limits_{k=0}^{n}2^{k}\left(\frac{1}{4^{k}} \lambda^{2}C_\mathrm{PI} + O\left(\frac{\lambda^{4}}{16^{k}}\right)\right)

\\&=\sum\limits_{k=0}^{n}\frac{1}{2^{k}} \lambda^{2}C_\mathrm{PI} + O(\frac{\lambda^{4}}{8^{k}})

\\&=2\lambda^{2}C_\mathrm{PI} - \frac{\lambda^{2}C_\mathrm{PI}}{2^{n}} + O(\frac{\lambda^{4}}{8^{k}})

\end{align}$$
Then, 
$$\begin{align}
m(\lambda) 
&\le \prod\limits_{k=0}^{n} \left(\frac{1}{1- \frac{1}{4^{k}} \lambda^{2} C_\mathrm{PI}}\right)^{2^{k}} m^{2^{n}} \left(\frac{\lambda}{2^{n+1}}\right)\quad\quad\text{Let n}\to\infty

\\&\le\exp(2\lambda^{2}C_\mathrm{PI}) 
\end{align}$$

**e.g.** Let $g(\mathbf{x}) = \Vert\mathbf{x}\Vert_{2}$, $\mathbf{x} \in \mathbb{R}^{d}$. We have, w.h.p. $| \Vert\mathbf{x}\Vert_{2} - \underbrace{\mathbb{E}[\Vert\mathbf{x}\Vert_{2}]}_{\Theta(\sqrt{d})} | \lesssim \underbrace{\sqrt{C_{\mathrm{PI}}}}_{\text{dim-independent}}$
Nearly all mass are in a thin shell.

Following similar arguments, we can prove that LSI $\Rightarrow$ sub-Gaussian concentration.
(and therefore PI ${\nRightarrow}$ LSI)

## Stroock-Holley perturbation principle
**Theorem**. If $\pi$ satisfies LSI/PI with constant $C_\mathrm{LSI}$/$C_\mathrm{PI}$. For $h(\mathbf{x})$ satisfying $\sup\limits_{\mathbf{x}} \Vert h(\mathbf{x})\Vert \le B$, Let distribution $\mu$ satisfies
$$\mu(\mathbf{x}) = \frac{\pi(\mathbf{x})\exp(h(\mathbf{x}))}{\int \pi(\mathbf{s})\exp(h(\mathbf{s}))\mathrm{d}\mathbf{s}}$$
Then $\mu$ satisfies LSI/PI with constant $e^{2B}C_\mathrm{LSI}$/$e^{2B}C_\mathrm{PI}$.

**Proof**. Note that $e^{-B} \le \frac{\mu(\mathbf{x})}{\pi(\mathbf{x})} \le e^{B}$
For PI,
$$\begin{align}
\mathrm{Var}_{\mu}(f(\mathbf{x})) &= \inf\limits_{m \in \mathbb{R}} \mathbb{E}_{\mu}[|f(\mathbf{x}) - m|^{2}] 
\\&\le\inf\limits_{m\in\mathbb{R}} e^{B}\mathbb{E}_{\pi}[|f(\mathbf{x}) - m|^{2}] 
\\&\le e^{B} C_\mathrm{PI} \mathbb{E}_{\pi}[\Vert\nabla f\Vert^{2}]
\\&\le e^{2B}C_\mathrm{PI} \mathbb{E}_{\mu}[\Vert\nabla f\Vert^{2}]
\end{align}$$
For LSI, 
$$\begin{align}

\mathbb{E}_{\mu}[f\log f] - \mathbb{E}_{\mu}[f]\mathbb{E}_{\mu}[\log f] 
&= \inf\limits_{t\ge 0} \mathbb{E}_{\mu}\left[f\log \frac{f}{t}-f+t\right]

\\&\le\inf\limits_{t\ge 0} e^{B} \mathbb{E}_{\pi} \left[f\log \frac{f}{t}-f+t\right]

\\&\le 4e^{B}C_\mathrm{LSI}\mathbb{E}_{\pi}[\Vert\nabla f\Vert^{2}]

\\&\le 4e^{2B}C_\mathrm{LSI}\mathbb{E}_{\mu}[\Vert\nabla f\Vert^{2}]
\end{align}$$

**e.g.** For non-convex, L-smooth function $f$, $|\nabla f(x) - \nabla f(y)|\le L|x-y|$, for $\forall x, y$. Let $x^{\star}$ be a stationary point.
Suppose that function is strongly convex ($\nabla^{2}f \succeq \lambda \mathbb{I}_{d}$) for $x \notin \mathbb{B}(x^{\star}, R)$, only smooth for $x \in \mathbb{B}(x^{\star}, R)$.
We can construct function $\bar{f}$ satisfies $\begin{cases} \text{equals to }f,  & \text{, outside }\mathbb{B}(x^{\star}, R)\\ |\bar{f} - f|\le LR^{2} & \text{, inside }\mathbb{B}(x^{\star}, R)\end{cases}$
For $\pi \propto \exp(-f)$, $C_\mathrm{LSI}(\pi)\le C_\mathrm{LSI}(\exp(-f))\cdot\exp(2LR^{2})\le \frac{\exp(2LR^{2})}{\lambda}$

**Remark**: This bound is near-tight in worst case.
**e.g.** For double-well potential function $f$, $\pi\propto\exp( \frac{-f}{\epsilon} )$. 
Its (rescaled) Langevin diffusion $\mathrm{d}\mathbf{X}_{t} = -\nabla f(\mathbf{X}_{t})\mathrm{d}t + \sqrt{2\epsilon}\mathrm{d}\mathbf{B}_{t}$
Escape time $\tau \approx \exp(\frac{h}{\epsilon})$, where $h$ represents energy barrier height.

### Poincare constant under non-convexity
**Theorem**.(Bakry et. al.) PI holds true under:
(a). $\langle \mathbf{x}, \nabla f(\mathbf{x})\rangle \ge \alpha\Vert\mathbf{x}\Vert$, for $x \notin \mathbb{B}(0, R)$
(b). $a\Vert\nabla f(\mathbf{x})\Vert^{2} - \Delta f(\mathbf{x})\ge C$, for $x \notin \mathbb{B}(0, R)$

**Proof**. 
(i). Within $\mathbb{B}(0, R)$, PI is satisfied. (Use Stroock-Holley perturbation principle)
$$\mu(\mathbf{x}) = \frac{\pi(\mathbf{x}) \mathbb{1}_{\mathbb{B}(0, R)}(\mathbf{x})}{\int \pi(\mathbf{s})\mathbb{1}_{\mathbb{B}(0, R)}(\mathbf{s})\mathrm{d}\mathbf{s}}$$
(ii). If Lyapunov function $W$ satisfies $\mathcal{A}W(\mathbf{x}) \le -\theta W(\mathbf{x}) + b \mathbb{1}_{\mathbb{B}(0, R)}(\mathbf{x})$, then PI is satisfied. ($\theta \lt 0$, $b \lt 0$).
(iii). Verify the condition, $W(\mathbf{x}) = \exp(\gamma\Vert\mathbf{x}\Vert_{2})$ for $\gamma$ sufficiently small.

**Proof for (ii)**. for function f satisfying $\mathbb{E}_{\pi}[f] = 0$
$$\begin{align}
\int f^{2}(\mathbf{x}) \mathrm{d}\pi(\mathbf{x}) &\le 
\underbrace{\int \frac{-\mathcal{A}W}{\theta W}f^{2}\mathrm{d}\pi(\mathbf{x})}
_{\le\int \frac{|\nabla f|^{2}}{\theta}\mathrm{d}\pi(\mathbf{x})}
+ \underbrace{\int \frac{b}{\theta W} \mathbb{1}_{\mathbb{B}(0, R)}(\mathbf{x}) f^{2}\mathrm{d}\pi(\mathbf{x})}_{\text{PI within }\mathbb{B}(0, R)}

\end{align}$$
**KLS conjecture**. Assume $-\log\pi$ convex, $\mathbb{E}_{\pi}[\mathbf{X}] = 0$, $\mathbb{E}_{\pi}[\mathbf{X}\mathbf{X}^{\top}] = \mathbb{I}_{d}$, then $C_\mathrm{PI}(\pi)$ is dim-free.
Chen(2021). $\forall \epsilon \gt 0$, $C_\mathrm{PI}(\pi)$ grows slower that $O(d^{\epsilon})$

## Applications to Non-Convex Optimization

What does stationary distribution give us?
Consider a diffusion process $\mathrm{d}\mathbf{X}_{t} = -\nabla f(\mathbf{X}_{t})\mathrm{d}t + \sqrt{\frac{2}{\beta}}\mathrm{d}\mathbf{B}_{t}$, $\beta \gt 0$. Its stationary distribution $\pi(\mathbf{x})\propto\exp(-\beta f(\mathbf{x}))$ 
Goal: To analyze $\mathbb{E}_{\pi}[f(\mathbf{X})] - f_\mathrm{min} \le \text{?}$
$$\begin{align}
\int \pi(\mathbf{x}) f(\mathbf{x}) \mathrm{d}\mathbf{x}
&= \int\pi(\mathbf{x}) \left(
-\frac{1}{\beta} \log \pi(\mathbf{x}) - \frac{1}{\beta} \mathbf{z}_{\beta}
\right) \mathrm{d}\mathbf{x}

\\&= \frac{1}{\beta} \underbrace{\int \pi(\mathbf{x})\log \frac{1}{\pi(\mathbf{x})} \mathrm{d}\mathbf{x}}_{\text{differential entropy }\le {\text{ D.E. of Gaussian}}}
- \frac{1}{\beta}\log \mathbf{z}_{\beta}

\\&\le \frac{1}{\beta} \frac{d}{2} \log(\mathbb{E}_{\pi}[\Vert\mathbf{X}\Vert^{2}]) - \frac{1}{\beta}\log\int\exp(-\beta f(\mathbf{z}))\mathrm{d}\mathbf{z}

\\
\text{Second term} &= f_\mathrm{min} - \frac{1}{\beta}\log \int \exp(-\beta(f(\mathbf{z}) - f_\mathrm{min}))\mathrm{d}\mathbf{z}
\\&\le f_\mathrm{min} - \frac{1}{\beta} \log \int \underbrace{\exp\left(-\frac{\beta L \Vert\mathbf{x} -\mathbf{x}^{\star}\Vert^{2}}{2}\right)}_{\text{Gaussian p.d.f.}} \mathrm{d}\mathbf{x}
\\&\le f_\mathrm{min} + \frac{d}{\beta}\log (\cdots)
\end{align}$$
Finally,
$$\begin{align}
\mathbb{E}_{\pi}[f(\mathbf{X})] - f_\mathrm{min} \le \frac{d}{\beta}\log(\cdots)
\end{align}$$
**Noise helps!**

# Lecture 4. Discretization of diffusion process, application to MCMC and stochastic optimization

## Discretization

**Forward-Euler method**. Take Langevin diffusion $\mathrm{d}\mathbf{X}_{t}=-\nabla f(\mathbf{X}_{t})\mathrm{d}t + \sqrt{2}\mathrm{d}\mathbf{B}_{t}$ with step size $\eta$.
Then, $\tilde{\mathbf{X}}_{(k+1)\eta} = \tilde{\mathbf{X}}_{k\eta} - \eta \nabla f(\tilde{\mathbf{X}}_{k\eta}) + \sqrt{2\eta} \mathbf{\xi}_{k}$, where $(\mathbf{\xi}_{0}, \mathbf{\xi}_{1}, \mathbf{\xi}_{2}, \cdots) \overset{\text{i.i.d}}{\sim} \mathcal{N}(0, \mathbb{I}_{d})$
Define a stochastic process $\mathrm{d}\tilde{\mathbf{X}}_{t} = -\nabla f(\tilde{\mathbf{X}}_{k\eta}) \mathrm{d}t + \sqrt{2}\mathrm{d}\mathbf{B}_{t}$, for $t \in [\tilde{\mathbf{X}}_{k\eta}, \tilde{\mathbf{X}}_{(k+1)\eta})$
Synchronous coupling: Drive $(\mathbf{X}_{t})_{t\ge 0}$ and $(\tilde{\mathbf{X}}_{t})_{t\ge 0}$ using the same BM.
Then, we get $\mathrm{d}(\mathbf{X}_{t} - \tilde{\mathbf{X}}_{t}) = -(\nabla f (\tilde{\mathbf{X}}_{k\eta}) - \nabla f(\mathbf{X}_{t}))\mathrm{d}t$

>[!note]
>In ML applications. Dependence on:
>(a). Total time
>(b). Problem dimension

$$\begin{align}
\frac{\mathrm{d}}{\mathrm{d}t} \mathbb{E}[\Vert \tilde{\mathbf{X}}_{t} - \mathbf{X}_{t} \Vert_{2}^{2}] 

&= -2\mathbb{E}[\langle
\tilde{\mathbf{X}}_{t} - \mathbf{X}_{t}, \nabla f (\tilde{\mathbf{X}}_{k\eta}) - \nabla f(\mathbf{X}_{t})
\rangle]

\\&\le -2\mathbb{E}[\langle
\tilde{\mathbf{X}}_{t} - \mathbf{X}_{t}, \nabla f (\tilde{\mathbf{X}}_{t}) - \nabla f(\mathbf{X}_{t})
\rangle]
-2\mathbb{E}[\langle
\tilde{\mathbf{X}}_{t} - \mathbf{X}_{t}, \nabla f (\tilde{\mathbf{X}}_{k\eta}) - \nabla f(\tilde{\mathbf{X}}_{t})
\rangle]

\\&\le -2\lambda\mathbb{E} [\Vert \tilde{\mathbf{X}}_{t} - \mathbf{X}_{t} \Vert_{2}^{2}] + 2 \sqrt{
\mathbb{E} [\Vert \tilde{\mathbf{X}}_{t} - \mathbf{X}_{t} \Vert_{2}^{2}]
} \cdot \sqrt{
\mathbb{E} [\Vert \nabla f (\tilde{\mathbf{X}}_{k\eta}) - \nabla f(\tilde{\mathbf{X}}_{t}) \Vert_{2}^{2}]
}

\end{align}$$
Under smoothness,
$$\begin{align}

\mathbb{E} [\Vert \nabla f (\tilde{\mathbf{X}}_{k\eta}) - \nabla f(\tilde{\mathbf{X}}_{t}) \Vert_{2}^{2}]
&\le L^{2} \mathbb{E} [\Vert \tilde{\mathbf{X}}_{k\eta} - \tilde{\mathbf{X}}_{t} \Vert_{2}^{2}]
\\

\tilde{\mathbf{X}}_{t} - \tilde{\mathbf{X}}_{k\eta} &=
\underbrace{\int_{k\eta}^{t} \nabla f(\tilde{\mathbf{X}}_{s}) \mathrm{d}s }_{O(\eta \sqrt{d})}
+ \underbrace{\sqrt{2}\int_{k\eta}^{t}\sqrt{2}\mathrm{d}\mathbf{B}_s}_{O(\sqrt{\eta d})}

\end{align}$$
Due to, 
$$\begin{align}
\mathbb{E}[\Vert\text{First term}\Vert_{2}^{2}] &\le
(t-k\eta) \int_{k\eta}^{t} \mathbb{E}[
\Vert\nabla f(\tilde{\mathbf{X}}_{s})\Vert_{2}^{2} \mathrm{d}s
] \\&\le (t-k\eta) \int_{k\eta}^{t} L^{2} \mathbb{E}[
\Vert \tilde{\mathbf{X}}_{s} - \mathbf{X}^{\star} \Vert_{2}^{2}
] \mathrm{d}s
\\&= O(\eta^{2} d)

\\ \mathbb{E}[\Vert\text{Second term}\Vert_{2}^{2}] &=
2d(t-k\eta)
\\&= O(\eta d)
\end{align}$$
Substituting back, $\Phi_{t} = \mathbb{E}[\Vert\tilde{\mathbf{X}}_{t} - \mathbf{X}_{t}\Vert^{2}]$
$$\begin{align}
\frac{\mathrm{d}\Phi_{t}}{\mathrm{d}t} &\le
-\lambda\Phi_{t} + \sqrt{\Phi_{t}\cdot C\cdot d\eta}
\\&\le -\lambda \Phi_{t} + \frac{\lambda}{2} \Phi_{t} + \frac{2C}{\lambda} d\eta
\\&= -\frac{\lambda}{2} \Phi_{t} + C^{\prime}\eta d
\end{align}$$
Solve for recursion,
$$\Phi_{t} \le \exp\left(-\frac{\lambda t}{2}\right) \Phi_{0} + \frac{2C^{\prime}}{\lambda} \eta d \le \epsilon^{2}$$
We need $\eta \le \Theta(\frac{\epsilon^{2}}{d})$, $t \ge \frac{4}{\lambda} \log( \frac{1}{\epsilon} )$. Number of steps: $O( \frac{d}{\epsilon^{2}} \log(\frac{1}{\epsilon}) )$

**Theorem**. Under above parameter setup, we have
$$\mathcal{W}_{2}(\tilde{\pi}_{n}, \pi^{\star}) \le \epsilon$$
**Remark**. 
- We did not optimize dependence or condition number $\kappa = \frac{l}{\lambda}$. (may be improved)
- "Unadjusted Langevin Algorithm"

Faster convergence possible under Metropolis adjustment.
Brief description of Metropolis-Hastings:
- ("Reversible") Fact, if an MC satisfies $p(y|x) \pi(x) = p(x|y) \pi(y)$. Then $\pi$ is a stationary distribution.
- MH: Accept or reject proposal.
Let $p$ be the proposal transition probability. For each step, generate $Y_{k} \sim p(\cdot|X_{k})$.
Then, accept $Y_{k}$ to proceed with probability $min( 1, \frac{p(X_{k}|Y_{k}) \pi(Y_{k})}{p(Y_{k}|X_{k}) \pi(X_{k})} )$.
We get adjusted Markov Chain, which satisfies
$$\begin{align*}
q(y|x) \pi(x) &= q(x|y) \pi(y)
\end{align*}$$
$\Rightarrow$ MALA(Metropoli-adjusted Langevin Algorithm)
- With $O(d\log(\frac{1}{\epsilon}))$ iterations, we get $d_\mathrm{TV}(\tilde{\pi}_{n}^\mathrm{MALA}, \pi^{\star}) \le \epsilon$. (may be sensitive to $\eta$ e.t.c.)

### Underdamped Langevin
$$\begin{cases}
\mathrm{d}\mathbf{v}_{t} = -\gamma \mathbf{v}_{t}\mathrm{d}t - u \nabla f(\mathbf{x}_{t})\mathrm{d}t + \sqrt{2\gamma u}\mathrm{d}\mathbf{B}_{t} \\
\mathrm{d}\mathbf{x}_{t} = \mathbf{v}_{t} \mathrm{d}t
\end{cases}$$
Discretization, for $t \in [k\eta, (k+1)\eta)$
$$\begin{cases}
\mathrm{d}\tilde{\mathbf{v}}_{t} = -\gamma \tilde{\mathbf{v}}_{t}\mathrm{d}t - u \nabla f(\tilde{\mathbf{x}}_{k\eta})\mathrm{d}t + \sqrt{2\gamma u}\mathrm{d}\mathbf{B}_{t} \\ 

\mathrm{d}\tilde{\mathbf{x}}_{t} = \tilde{\mathbf{v}}_{t} \mathrm{d}t
\end{cases}$$
Ornstein-Uhlenbeck process:
Given $(\tilde{\mathbf{x}}_{k\eta}, \tilde{\mathbf{v}}_{k\eta})$, the conditional distribution of $(\tilde{\mathbf{x}}_{k\eta}, \tilde{\mathbf{v}}_{k\eta})$ is normal, with closed-form formulae for mean and variance.
Use this at update algo.

Let Lyapunov function
$$\Phi_{t} = \mathbb{E} \left[\left\Vert \begin{pmatrix} 1 & 1 \\ 0 & 1\end{pmatrix} \begin{pmatrix} \tilde{\mathbf{x}}_{t}-{\mathbf{x}}_{t} \\ \tilde{\mathbf{v}}_{t}-{\mathbf{v}}_{t}\end{pmatrix}\right\Vert_{2}^{2} \right]$$
Let $\mathbf{z}_{t} = \tilde{\mathbf{x}}_{t}-{\mathbf{x}}_{t}$, $\mathbf{y}_{t} =\tilde{\mathbf{v}}_{t}-{\mathbf{v}}_{t}$
$$\begin{align*}
\frac{\mathrm{d}\Phi_{t}}{\mathrm{d}t} &= 
2\mathbb{E}\left[
\begin{pmatrix}
\mathbf{z}_{t} + \mathbf{y}_{t} &  \mathbf{y}_{t} 
\end{pmatrix}
\begin{pmatrix}
(1-\gamma)\mathbf{y}_{t} - u \left(\nabla f(\tilde{\mathbf{x}}_{k\eta}) - \nabla f(\mathbf{x}_{t})\right) \\ -\gamma\mathbf{y}_{t}  - u \left(\nabla f(\tilde{\mathbf{x}}_{k\eta}) - \nabla f(\mathbf{x}_{t})\right)
\end{pmatrix}
\right]\\
&= -(\text{something nice}) + u \mathbb{E}\left[
\begin{pmatrix}
\mathbf{z}_{t} + \mathbf{y}_{t} &  \mathbf{y}_{t} 
\end{pmatrix}
\begin{pmatrix}
- \left(\nabla f(\tilde{\mathbf{x}}_{k\eta}) - \nabla f(\tilde{\mathbf{x}}_{t})\right) \\ - \left(\nabla f(\tilde{\mathbf{x}}_{k\eta}) - \nabla f(\tilde{\mathbf{x}}_{t})\right)
\end{pmatrix}
\right]\\
&\le -\frac{\lambda}{2L} \Phi_{t} + u \sqrt{\Phi_{t} \cdot \mathbb{E}[\Vert
\nabla f(\tilde{\mathbf{x}}_{k\eta}) - \nabla f(\tilde{\mathbf{x}}_{t})
\Vert^{2}]}\\
\end{align*}$$
Due to
$$\begin{align*}
\mathbb{E}[\Vert
\nabla f(\tilde{\mathbf{x}}_{k\eta}) - \nabla f(\tilde{\mathbf{x}}_{t})
\Vert^{2}] &\le L^{2} \mathbb{E}[\Vert
\tilde{\mathbf{x}}_{k\eta} - \tilde{\mathbf{x}}_{t}
\Vert^{2}]\\
\tilde{\mathbf{x}}_{t} &= \tilde{\mathbf{x}}_{k\eta} + \int_{\eta k}^{t} \tilde{\mathbf{v}}_{t} \mathrm{d}t
\end{align*}$$
So 
$$\begin{align*}
\mathbb{E}[\Vert
\tilde{\mathbf{x}}_{k\eta} - \tilde{\mathbf{x}}_{t}
\Vert^{2}] &\le 
\eta^{2} \max\limits_{k\eta\le t \le (k+1)\eta} \mathbb{E}[\Vert\tilde{\mathbf{v}}_{t}\Vert^{2}]
\end{align*}$$
Similar to Langevin Algorithm, $\mathbb{E}[\Vert\tilde{\mathbf{v}}_{t}\Vert^{2}] \le O(d)$
Finally, we get
$$\begin{align*}
\Phi_{t} &\le \exp\left(- \frac{\lambda}{2L} t\right)\Phi_{0} + O(\eta^{2}d)
\end{align*}$$
If we want $\mathcal{W}_{2} (\tilde{\pi}_{t}, \pi^{\star}) \le \epsilon$, we need $\eta \le O(\frac{\epsilon}{\sqrt{d}})$, $t \ge \Omega(\frac{L}{\lambda} \log(\frac{1}{\epsilon}))$
Iteration complexity $n \asymp \frac{\sqrt{d}}{\epsilon} \log(\frac{1}{\epsilon})$

**Remark**. some further improvements
- (Shen & Lee, 2019) Randomized midpoint, $O(d^{\frac{1}{3}})$
- (Mou et. al., 2019) High-order Langevin (Using integration orade, $O(d^{\frac{1}{4}})$
- (Song, Lee & Venpala, 2019) With "nice data" (Specialized to Bayesian logistic regression), $O(\mathrm{polylog}(d))$
- (Zhang, 2015) Randomized midpoint, $O(d^{\frac{1}{4}})$
Still open: $\mathrm{polylog}(1/\epsilon)$, sub-linear in $d$ for cold start.

**How about non-convex?**
For diffusion process
$$\begin{align*}
\mathrm{d}\mathbf{x}_{t} &= -\nabla f(\mathbf{x}_{t})\mathrm{d}t + \sqrt{2}\mathrm{d}\mathbf{B}_{t}\\
\mathrm{d}\tilde{\mathbf{x}}_{t} &= -\nabla f(\tilde{\mathbf{x}}_{k\eta})\mathrm{d}t + \sqrt{2}\mathrm{d}\mathbf{B}_{t}
\end{align*}$$
Idea 1: Still use synchronous coupling to estimate
$$\begin{align*}
\frac{\mathrm{d}}{\mathrm{d}t} \mathbb{E}[\Vert\tilde{\mathbf{x}}_{t} - \mathbf{x}_{t}\Vert^{2}] 
&\le -2\mathbb{E}[\langle \mathbf{x}_{t}-\tilde{\mathbf{x}}_{t},  \nabla f(\mathbf{x}_{t})-\nabla f(\tilde{\mathbf{x}}_{k\eta}) \rangle]\\
&\le 2L \cdot \mathbb{E} [\Vert \mathbf{x}_{t}-\tilde{\mathbf{x}}_{t}\Vert \cdot\Vert  \mathbf{x}_{t}-\tilde{\mathbf{x}}_{k\eta} \Vert] \\
&\le 2L\cdot \mathbb{E}[\Vert \mathbf{x}_{t}-\tilde{\mathbf{x}}_{t}\Vert^{2}] + 2L \mathbb{E}[\Vert \mathbf{x}_{t}-\tilde{\mathbf{x}}_{t}\Vert \Vert \tilde{\mathbf{x}}_{t}-\tilde{\mathbf{x}}_{k\eta}\Vert]\\
&\le 2L\cdot \mathbb{E}[\Vert \mathbf{x}_{t}-\tilde{\mathbf{x}}_{t}\Vert^{2}] + O(\eta d)
\end{align*}$$
Use Gronwall inequality, 
$$\begin{align*}
\mathbb{E}[\Vert\tilde{\mathbf{x}}_{t} - \mathbf{x}_{t}\Vert^{2}]  &\le O(\eta d)\cdot\exp(2Lt)
\end{align*}$$
This is unavailable for ODEs, but the noise may help!

**Solution 1**. pathwise KL divergence
**Radon-Nikodym derivative**. Let $\mathbb{P}^{(1)}$, $\mathbb{P}^{(2)}$ be two probability distribution on $\mathbf{X}$.
$\frac{\mathrm{d}\mathbb{P}^{(1)}}{\mathrm{d}\mathbb{P}^{(2)}}$ as a random variable, we have
$$\mathbb{E}\left[f(\mathbf{X}^{(2)}) \frac{\mathrm{d}\mathbb{P}^{(1)}}{\mathrm{d}\mathbb{P}^{(2)}}\right] = \mathbb{E}[f(\mathbf{X}^{(1)})]$$
Underlying idea: $p^{(1)}$, $p^{(2)}$ are densities, 
$$\frac{\mathrm{d}\mathbb{P}^{(1)}}{\mathrm{d}\mathbb{P}^{(2)}} = \frac{p^{(1)}}{p^{(2)}}(\mathbf{X}^{(2)})$$
where $X^{(2)} \sim \mathbb{P}^{(2)}$.

$$\mathcal{D}_\mathrm{KL}(\mathbb{P}^{(1)}\Vert\mathbb{P}^{(2)}) = -\mathbb{E}\left[ \log \frac{\mathrm{d}\mathbb{P}^{(2)}}{\mathrm{d}\mathbb{P}^{(1)}} \right]$$

**Definition**.
$\tilde{\mathbb{P}}_{[0, T]}$: Law of $(\tilde{\mathbf{X}}_{t}: t\in [0,T])$
${\mathbb{P}}_{[0, T]}$: Law of $({\mathbf{X}}_{t}: t\in [0,T])$

**Analogy**: discrete-time version.
Let
$$\begin{align*}
\mathbf{X}_{t+1}^{(1)}&= \mathbf{h}^{(1)}(\mathbf{X}^{(1)}_{1}, \cdots, \mathbf{X}^{(1)}_{t}) + \mathbf{\xi}_{t+1}\\
\mathbf{X}_{t+1}^{(2)}&= \mathbf{h}^{(2)}(\mathbf{X}^{(2)}_{1}, \cdots, \mathbf{X}^{(2)}_{t}) + \mathbf{\xi}_{t+1}
\end{align*}$$
where $\xi_{t} \sim \mathcal{N}(0, \mathbb{I}_{d})$.

$$\begin{align*}
P_{\mathbf{X}_{t+1}^{(1)}}(\mathbf{X}|\mathbf{X}^{(1)}_{1}, \cdots, \mathbf{X}^{(1)}_{t})
&= \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{1}{2}\Vert\mathbf{X}-\mathbf{h}^{(1)}(\mathbf{X}^{(1)}_{1}, \cdots, \mathbf{X}^{(1)}_{t})\Vert^{2}\right)\\
P_{\mathbf{X}_{t+1}^{(2)}}(\mathbf{X}|\mathbf{X}^{(2)}_{1}, \cdots, \mathbf{X}^{(2)}_{t})
&= \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{1}{2}\Vert\mathbf{X}-\mathbf{h}^{(1)}(\mathbf{X}^{(2)}_{1}, \cdots, \mathbf{X}^{(2)}_{t})\Vert^{2}\right)
\end{align*}$$
Then,
$$\begin{align*}
\frac{P_{\mathbf{X}_{t+1}^{(2)}}}{P_{\mathbf{X}_{t+1}^{(1)}}} &= 
\exp\left(\langle \mathbf{X},  \mathbf{h}^{(2)} - \mathbf{h}^{(1)}\rangle
- \frac{1}{2} \Vert\mathbf{h}^{(2)}\Vert^{2} + \frac{1}{2} \Vert\mathbf{h}^{(1)}\Vert^{2}\right) \\
&= \exp \left(\langle \mathbf{X}-\mathbf{h}^{(1)}, \mathbf{h}^{(2)}-\mathbf{h}^{(1)} \rangle - \frac{1}{2}\Vert \mathbf{h}^{(1)}-\mathbf{h}^{(2)} \Vert^{2}\right)

\end{align*}$$
Finally, 
$$\begin{align*}
\frac{\mathrm{d}\mathbb{P}^{(2)}}{\mathrm{d}\mathbb{P}^{(1)}}=
\exp(
\sum\limits_{i=1}^{n} \langle \xi_{i}, \mathbf{h}^{(2)}(\mathbf{X}^{(1)}_{1}, \cdots, \mathbf{X}^{(1)}_{i-1}) - \mathbf{h}^{(1)}(\mathbf{X}^{(1)}_{1}, \cdots, \mathbf{X}^{(1)}_{i-1}) \rangle \\- \frac{1}{2} \Vert \mathbf{h}^{(1)}(\mathbf{X}^{(1)}_{1}, \cdots, \mathbf{X}^{(1)}_{i-1}) - \mathbf{h}^{(2)}(\mathbf{X}^{(1)}_{1}, \cdots, \mathbf{X}^{(1)}_{i-1}) \Vert^{2}
)
\end{align*}$$

**Theorem**. (Girsanov) Suppose $\mathrm{d}\mathbf{X}_{t}^{(i)} = \mathbf{g}^{(i)}(\mathbf{X}^{(i)}_{[0,T]})\mathrm{d}t + \mathrm{d}\mathbf{B}_{t}$, $i \in \{1, 2\}$
Calculate, 
$$
\frac{\mathrm{d}\mathbb{P}_{[0,T]}^{(1)}}{\mathrm{d}\mathbb{P}_{[0,T]}^{(2)}} = \exp\left(
\int_{0}^{T}(\mathbf{g}^{(1)}(\mathbf{X}^{(2)}_{[0,T]})-\mathbf{g}^{(2)}(\mathbf{X}^{(2)}_{[0,T]}))^{\top}\mathrm{d}\mathbf{B}_{t} - \frac{1}{2}\int_{0}^{T} \Vert \mathbf{g}^{(1)}(\mathbf{X}^{(2)}_{[0,T]})-\mathbf{g}^{(2)}(\mathbf{X}^{(2)}_{[0,T]})\Vert^{2}\mathrm{d}t \right)
$$
Then, (under weak assumptions)
$$\begin{align*}
\mathcal{D}_\mathrm{KL} (\tilde{\mathbb{P}}_{[0,T]} \Vert \mathbb{P}_{[0,T]})
&=  -\mathbb{E}\left[ \log \frac{\mathrm{d}\mathbb{P}_{1}}{\mathrm{d}\mathbb{P}_{2}} \right] - \int_{0}^{T}\mathbb{E}[\Vert \nabla f(\mathbf{X}_{k\eta})-\nabla f(\mathbf{X}_{t}) \Vert^{2}]\mathrm{d}t\\
&\le L^{2}\int_{0}^{T} \underbrace{\mathbb{E}[\Vert \mathbf{X}_{t} - \mathbf{X}_{k\eta} \Vert^{2}] }_{\le O(d\eta)}\mathrm{d}t 
\end{align*}
$$
$$
\mathcal{D}_\mathrm{KL}(\hat{\pi}_{T}\Vert\pi_{T}) \le \mathcal{D}_\mathrm{KL} (\tilde{\mathbb{P}}_{[0,T]} \Vert \mathbb{P}_{[0,T]}) \le O(T\eta d)
$$
Pinsker+triangle inequality,
$$d_\mathrm{TV}(\hat{\pi}_{T}, \pi^{\star}) \le \exp\left(- \frac{T}{2C_\mathrm{LSI}}\right)+ O(\sqrt{\eta Td})$$
For $\epsilon$-close in TV, we need $\eta \lesssim \frac{\epsilon^{2}}{Td}$, $T \gtrsim C_\mathrm{LSI} \log(\frac{1}{\epsilon})$, Its complexity $n \asymp \frac{C_\mathrm{LSI}^{2}d}{\epsilon^{2}}\log(\frac{1}{\epsilon})$   
We get $O(\sqrt{\eta})$ numerical order, not tight! (compared to Euler method in ODE)

**Fokker-Planck equation method**. (Mou et. al., 2020)
$$\begin{align*}
\partial_{t}\pi_{t} &= \nabla\cdot(\pi_{t}\nabla f) + \Delta \pi_{t}
\\
\partial_{t}\tilde{\pi}_{t}
 &=  \nabla\cdot(\tilde{\pi}_{t}{\mathbb{E}[\nabla f(\tilde{\mathbf{x}}_{k\eta})|\tilde{\mathbf{x}}_{t}=\mathbf{x}]})
 + \Delta \hat{\pi}_{t}
\end{align*}
$$

# Lecture 5. Advanced topics in discretization and sampling, Score-based diffusion generative models

From last lecture, we have
$$\begin{align*}
\mathcal{D}_\mathrm{KL}(\hat{\mathbb{P}}_{[0, T]}\Vert \mathbb{P}_{[0,T]}) &\le O(\eta dT)
\end{align*}$$
Numerical order not tight enough.

Idea: compare $\hat{\pi}_{t}$ and $\pi_{t}$
By Fokker-Planck equation:
$$
\frac{\partial\pi_{t}}{\partial t} = \nabla\cdot (\pi_{t} \nabla f) + \Delta \pi_{t}
$$
And we have, 
$$\mathrm{d}\hat{\mathbf{x}}_{t} = -\nabla f(\hat{\mathbf{x}}_{k\eta})\mathrm{d}t + \sqrt{2}\mathrm{d}\mathbf{B}_{t}$$
Then,
$$\frac{\partial\hat{\pi}_{t}|_{k\eta}}{\partial t}
 = \nabla\cdot(\hat{\pi}_{t}|_{k\eta}\cdot\underbrace{\nabla f(\hat{\mathbf{x}}_{k\eta})}_{\substack{\text{seen as deterministic}\\ \text{since we condition on }\mathcal{F}_{k\eta}}})
 + \Delta \hat{\pi}_{t}|_{k\eta}
$$

From $\hat{\pi}_{t}|_{k\eta}$ to $\hat{\pi}_{t}$: integrate out $\hat{\mathbf{x}}_{k\eta}$,
$$
\frac{\partial\hat{\pi}_{t}(\mathbf{x})}{\partial t} = \nabla\cdot (\hat{\pi}_{t}(\mathbf{x}) \underbrace{\mathbb{E}\left[\nabla f(\hat{\mathbf{x}}_{k\eta})|\hat{\mathbf{x}}_{t}=\mathbf{x}\right]}_{=:\hat{\mathbf{b}}_{t}(\mathbf{x})}
)  + \Delta \hat{\pi}_{t}
$$
Let $\mathbf{b}(\mathbf{x}) \coloneqq \nabla f(\mathbf{x})$ for simplicity,
$$\begin{align*}
\partial_{t}\hat{\pi}_{t} &= \nabla\cdot (\hat{\pi}_{t} \hat{\mathbf{b}}_{t}) + \Delta \hat{\pi}_{t}\\
\partial_{t} \pi_{t} &= \nabla\cdot (\pi_{t} \mathbf{b}) + \Delta \pi_{t}
\end{align*}$$
Then, we calculate,
$$\begin{align*}
\frac{\mathrm{d}}{\mathrm{d}t}\mathcal{D}_\mathrm{KL}(\hat{\pi}_{t}\Vert\pi_{t})
&= \int \frac{\partial}{\partial t}\left(\hat{\pi}_{t}\log \frac{\hat{\pi}_{t}}{\pi_{t}} \right)\mathrm{d}\mathbf{x}\\

&= \int (\nabla\cdot (\hat{\pi}_{t} \hat{\mathbf{b}}_{t}) + \Delta \hat{\pi}_{t}) \log \frac{\hat{\pi}_{t}}{\pi_{t}} \mathrm{d}\mathbf{x} + {\int \pi_{t} 
\frac{\partial}{\partial t}\left( \frac{\hat{\pi}_{t}}{\pi_{t}}\right) \mathrm{d}\mathbf{x}} \\

&= \int (\nabla\cdot (\hat{\pi}_{t} \hat{\mathbf{b}}_{t}) + \Delta \hat{\pi}_{t}) \log \frac{\hat{\pi}_{t}}{\pi_{t}} \mathrm{d}\mathbf{x} +
\int  \pi_{t} \frac{\partial_{t}\hat{\pi}_{t}\pi _{t} - \hat{\pi}_{t}\partial_{t}\pi _{t}}{\pi_{t}^{2}} \mathrm{d}\mathbf{x}\\

&= \int\langle \mathbf{b}-\hat{\mathbf{b}}_{t}, \nabla\log \frac{\hat{\pi}_{t}}{\pi_{t}} \rangle \hat{\pi}_{t} \mathrm{d}\mathbf{x} - 
\int \Vert \nabla\log \frac{\hat{\pi}_{t}}{\pi_{t}} \Vert^{2}\hat{\pi}_{t}\mathrm{d}\mathbf{x}\\

&\le \int \Vert \mathbf{b}_{t}(\mathbf{x})-\hat{\mathbf{b}}_{t}(\mathbf{x})\Vert^{2}\hat{\pi}_{t}(\mathbf{x}) \mathrm{d}\mathbf{x}
\end{align*}$$
Previously, we have, $\mathbb{E}[\Vert \mathbf{b}(\hat{\mathbf{x}}_{t})-\hat{\mathbf{b}}(\hat{\mathbf{x}}_{k\eta})\Vert^{2}]$
Now we get, 
$$
\mathbb{E}[\Vert \mathbf{b}(\hat{\mathbf{x}}_{t})-
\mathbb{E}[\hat{\mathbf{b}}(\hat{\mathbf{x}}_{k\eta})|\hat{\mathbf{x}}_{t}]\Vert^{2}]
$$
**Is it better**?

Conceptual example: two extremes
- If $\hat{\mathbf{x}}_{k\eta}$ is deterministic, $\mathbb{E}[\Vert \mathbf{b}-\hat{\mathbf{b}}_{t} \Vert^{2}] = \Theta(\eta)$.
- If the density of $\hat{\mathbf{x}}_{k\eta}$ is "uniform", $\mathbb{E}[\Vert \mathbf{b}-\hat{\mathbf{b}}_{t} \Vert^{2}] = \Theta(\eta) \le O(\eta^{2})$ in the "best case".

Analysis based on this intuition:
**Step 1**. linear approximate to b.
$$\begin{align*}
\mathbf{b}(\hat{\mathbf{x}}_{k\eta}) &= \mathbf{b}(\hat{\mathbf{x}}_{t}) + \nabla\cdot\mathbf{b}(\hat{\mathbf{x}}_{t}) (\hat{\mathbf{x}}_{k\eta} - \hat{\mathbf{x}}_{t}) + O(\eta^{2})
\end{align*}$$
Only need to analyze (Condition on $\hat{\mathbf{x}}_{t}=\mathbf{x}$)
$$\begin{align*}
\mathbb{E}[\hat{\mathbf{x}}_{k\eta} | \hat{\mathbf{x}}_{t}] - \hat{\mathbf{x}}_{t}
&= \int \frac{\hat{\pi}_{k\eta}(\mathbf{y})\cdot\hat{p}_{t|k\eta}(\mathbf{x}|\mathbf{y})}{\hat{\pi}_{t}(\mathbf{x})} (\mathbf{y} - \mathbf{x})\mathrm{d}\mathbf{y}
\end{align*}$$
where $\hat{p}_{t|k\eta} \sim \mathcal{N}(\mathbf{y}-\eta b(\mathbf{y}), 2\eta \mathbb{I}_{d})$ density evaluated at $\mathbf{x}$.
Then, 
$$\begin{align*}
\mathbb{E}[\hat{\mathbf{x}}_{k\eta} | \hat{\mathbf{x}}_{t}] - \hat{\mathbf{x}}_{t}
&\approx \int \frac{\hat{\pi}_{k\eta}(\mathbf{y})\cdot\nabla_{y}\hat{p}_{t|k\eta}(\mathbf{x}|\mathbf{y}) }{\hat{\pi}_{t}(\mathbf{x})} \mathrm{d}\mathbf{y}
\end{align*}$$
Then,
$$\begin{align*}
\mathbb{E} [\Vert \mathbb{E}[\hat{\mathbf{x}}_{k\eta} | \hat{\mathbf{x}}_{t}] - \hat{\mathbf{x}}_{t}\Vert^{2}] &\le \eta^{2}\int\hat{\pi}_{k\eta}\Vert \nabla\log\hat{\pi}_{k\eta} \Vert^{2} \mathrm{d}\mathbf{x} + O(\eta^{2})
\end{align*}$$
**Intuition**: 
$$\hat{\pi}_{k\eta} 
\xrightarrow{\mathbf{x} \mapsto \mathbf{x}-\eta \mathbf{b}(\mathbf{x})}
\text{increase FI} \xrightarrow{\text{add }\mathcal{N}(0, \eta\mathbb{I}_{d})}
\text{decrease FI} \rightarrow \hat{\pi}_{(k+1)\eta}
$$
Then we conclude, 
$$\mathcal{D}_\mathrm{KL}(\hat{\pi}_{t}\Vert\pi_{t}) \le O(\eta^{2}T)$$
Wang and Li (2024). Extension to general diffusion

## Score-based generative models

**Goal**: Learn $p^{\star}$ from $\mathbf{x}_{1}, \mathbf{x}_{2}, \cdots, \mathbf{x}_{n} \sim p^{\star}$

**Idea 1**: Learn $\hat{\mathbf{s}}(\mathbf{x})\approx\nabla\log p^{\star}(\mathbf{x})$. Run diffusion process $\mathrm{d}\hat{\mathbf{x}}_{t}=-\hat{\mathbf{s}}(\mathbf{x}_{t}) \mathrm{d}t + \sqrt{2}\mathrm{d}\mathbf{B}_{t}$ (and its numerical simulation)
**Problem**: How to learn $\nabla\log p^{\star}(\mathbf{x})$? 
In regression problems, we learn $f^{\star}$ from $\mathbf{y}_{i} = \mathbf{f}(\mathbf{x}_{i}) + \mathbf{\epsilon}_{i}$, $\mathbb{E}[\mathbf{y}|\mathbf{x}] = \mathbf{f}(\mathbf{x})$, $\text{loss} = \mathbb{E}[\Vert \mathbf{Y}-\mathbf{f}(\mathbf{X}) \Vert^{2}]$.
Empirically, $\text{loss} = \frac{1}{n} \sum\limits_{i=1}^{n}(\mathbf{y}_{i} - \mathbf{f}(\mathbf{x}_{i}))^{2}$
In score-based generative models, if we define
$$\begin{align*}
\rm{loss}(\mathbf{s})&= \mathbb{E}[\Vert \nabla\log p^{\star}(\mathbf{X}) - \mathbf{s(\mathbf{X})} \Vert^{2}]\\
&= \underbrace{\mathbb{E}[\Vert\mathbf{s}(\mathbf{X})\Vert^{2}]}_{\text{easy to compute}} - 2\mathbb{E}[\langle \mathbf{s}(\mathbf{X}), \nabla \log p^{\star}(\mathbf{X}) \rangle] + \underbrace{\mathbb{E}[\Vert \nabla \log p^{\star}(\mathbf{X}) \Vert^{2}]}_{\text{independend of } \mathbf{s}}
\end{align*}$$
For the second term, 
$$\begin{align*}
2\mathbb{E}[\langle \mathbf{s}(\mathbf{X}), \nabla \log p^{\star}(\mathbf{X}) \rangle] &= 
\int \mathbf{s}(\mathbf{x})^{\top} \nabla\log p^{\star}(\mathbf{x}) p^{\star}(\mathbf{x}) \mathrm{d}\mathbf{x}\\
&= \int \mathbf{s}(\mathbf{x})^{\top} \nabla p^{\star}(\mathbf{x}) \mathrm{d}\mathbf{x}\\
&= -\int(\nabla\cdot \mathbf{s}(\mathbf{x}))p^{\star}(\mathbf{x})\mathrm{d}\mathbf{x}
\end{align*}$$
So we have, 
$$\mathrm{loss}(\mathbf{s}) = \mathbf{E}[\Vert\mathbf{s}(\mathbf{X})\Vert^{2} + 2(\nabla\cdot\mathbf{s}(\mathbf{x}))] + \text{something independent of } \mathbf{s}$$
Then, 
$$\text{empirical loss} = \frac{1}{n} \sum\limits_{i=1}^{n}(\Vert\mathbf{s}(\mathbf{x}_{i})\Vert^{2} + 2\nabla\cdot\mathbf{s}(\mathbf{x}_{i}))$$
**e.g.** If we know $s^{\star} = \nabla\log p^{\star} \in \mathcal{F}$, $\mathcal{F}$ is convex. We can solve $\hat{\mathbf{s}}$ using convex optimization.

Performance guarantees?
**e.g.** $\mathbb{E}[\Vert \nabla\log p^{\star}(\mathbf{X})-\mathbf{s}(\mathbf{X})\Vert^{2}] \le \epsilon_{\text{score}}^{2}$
$$
\mathcal{D}_\mathrm{KL}(\mathbb{P}_{[0, T]}^{\star} \Vert \hat{\mathbb{P}}_{[0,T]})
\le T\epsilon^{2}_{\text{score}} + \text{discretization error}\le O(\eta T d)
$$

**e.g.** $\mathbf{s}(\mathbf{x}) = \nabla \log q(\mathbf{x})$, $q \in \mathcal{Q}$
$$
I(p^{\star}\Vert\hat{q}) = \mathbb{E}_{p^{\star}}[\Vert \mathbf{s}(\mathbf{X})-\nabla\log p^{\star}(\mathbf{X})\Vert^{2}]
$$

### Learning score function of data distribution

**Goal**: Fit $\nabla\log p^{\star}(\mathbf{x})$ using function set $\mathcal{F}$
$$\mathbf{s} = \mathop{\arg\min}_{\mathbf{s}\in\mathcal{F}} \mathbb{E}[\Vert \nabla\log p^{\star}(\mathbf{X}) - \mathbf{s(\mathbf{X})} \Vert^{2}] $$
Empirically,
$$\mathbf{s} = \mathop{\arg\min}_{\mathbf{s}\in\mathcal{F}} \frac{1}{n} \sum\limits_{i=1}^{n}(\Vert\mathbf{s}(\mathbf{x}_{i})\Vert^{2} + 2\nabla\cdot\mathbf{s}(\mathbf{x}_{i})) $$
Suppose $\mathcal{F}=\{ \nabla\log q : q\in\mathcal{Q} \}$, 
$$L(\nabla\log q) = \mathbb{E}_{p^{\star}}[\Vert\nabla\log p^{\star} - \nabla\log q \Vert^{2}]$$
If $\forall q \in \mathcal{Q}$, $q$ satisfies LSI ($C_{0}$), then
$$\mathcal{D}_\mathrm{KL} (p^{\star} \Vert q) \le C_{0}L(\nabla\log q)$$
Q: How to control $L(\nabla\log q)$?

>[!Detour] ML Theory basics
>In general, we want to minimize $L(f) = \mathbb{E}[l(f;\mathbf{X})]$
>Empirically, we calculate
>$$\hat{f}_{n} = \mathop{\arg\min}_{f\in\mathcal{F}} \underbrace{\frac{1}{n} \sum\limits_{i=1}^{n}l(f; \mathbf{X}_{i})}_{L_{n}(f)}$$
>How about $L(\hat{f}_{n})$?
>- $\forall f$, $\mathbb{E}[L_{n}(f)] = L(f)$,
>- $\hat{f}_{n}$ minimizes $L_{n}$,
>- $\mathbb{E}[L_{n}(\hat{f}_{n})] {\color{red}{\neq}} L(f^{\star})$

Let $f^{\star} = \mathop{\arg\min}_{f \in \mathcal{F}} L(f)$, 
$$\begin{align*}
L(\hat{f}_{n}) - L(f^{\star}) &= L(\hat{f}_{n}) - L_{n}(\hat{f}_{n}) + 
\underbrace{L_{n}(\hat{f}_{n}) - L_{n}(f^{\star})}_{\le 0} + 
\underbrace{L_{n}(f^{\star}) - L(f^{\star})}_{\substack{\text{easily obtained}\\ \text{from concentration}}}\\
&\le \sup_{f\in\mathcal{F}}| L(f) - L_{n}(f) |
\end{align*}$$
Need uniform convergence. Depending on complexity of $\mathcal{F}$
**e.g.** Suppose $|\mathcal{F}| \lt +\infty$
$$\begin{align}

\mathbb{P}\left(\sup_{f\in\mathcal{F}} |L_{n}(f) - L(f)|\ge\epsilon \right)
&\le |\mathcal{F}|\sup_{f\in\mathcal{F}}\mathbb{P}(|L_{n}(f) - L(f)|\ge\epsilon)\\
&\le |\mathcal{F}|\exp(-Cn\epsilon^{2})
\end{align}$$
If we want $\mathbb{P}\left(\sup_{f\in\mathcal{F}} |L_{n}(f) - L(f)|\ge\epsilon \right) \le \delta$, we can achieve
$$\epsilon_{n} = C\sqrt{ \frac{\log\left(\frac{|\mathcal{F}|}{\delta}\right)}{n} }$$

**e.g.** For infinite $\mathcal{F}$, use discrete approximation

However, relax to $\sup_{f\in\mathcal{F}}|\cdots|$ can be overly conservative.

Using **localization**: consider $\sup$ over $\mathcal{F}\cap\mathbb{B}(f^{\star}, r_{n})$

**Application to generative models**
Suppose $q^{\star} \in \mathcal{Q}$, any distribution in $\mathcal{Q}$ satisfies LSI ($C_{0}$), then
$$\mathcal{D}_\mathrm{KL} (p^{\star} \Vert \hat{q}_{n}) \le C_{0} \cdot\mathrm{complexity}_{n}(\mathcal{F})$$
where $\mathcal{F} = \{ \Vert \nabla\log q \Vert^{2} + 2 \Delta \log q : q \in \mathcal{Q} \}$
- Isoperimetry is unavoidable for this model

# Lecture 6. Denoising diffusion generative models, discretization and learning of diffusion models

## Denoising diffusion model

**Forward process** (add noise to image)
Let $\mathbf{X}_{0} \sim p^{\star}$. Run OU process $\mathrm{d}\mathbf{X}_{t} = -\mathbf{X}_{t}\mathrm{d}t + \sqrt{2} \mathrm{d}\mathbf{B}_{t}$, $t \in [0,T]$.
Converging to $\mathcal{N}(0, \mathbb{I}_{d})$ exponentially fast, independent of isoperimetry of $p^{\star}$.

We only need marginal distribution to be close.
Use Fokker-Planck equation:
$$\begin{align*}
\pi_{0} &= p^{\star}\\
\partial_{t}\pi_{t} &= \nabla\cdot(\mathbf{X}_{t}\pi_{t}) + \Delta\pi_{t}
\end{align*}$$
We want to reverse in time, let $\overleftarrow{\pi}_{t}\coloneqq\pi_{T-t}$, for $t \in [0,T]$, satisfying
$$
\begin{align*}
\partial_{t} \overleftarrow{\pi}_{t} &= -\nabla\cdot(\mathbf{X}\overleftarrow{\pi}_{t}) - \Delta \overleftarrow{\pi}_{t} \\
&= \Delta \overleftarrow{\pi}_{t} + (-\nabla\cdot(\mathbf{X} \overleftarrow{\pi}_{t}) - 2\Delta \overleftarrow{\pi}_{t})\\
&= \Delta \overleftarrow{\pi}_{t} + (-\nabla\cdot[(\mathbf{X} + 2\nabla\log \overleftarrow{\pi}_{t})\overleftarrow{\pi}_{t}])
\end{align*}
$$
This is Fokker-Planck of another diffusion,
$$\mathrm{d} \overleftarrow{\mathbf{X}}_{t} = (\overleftarrow{\mathbf{X}}_{t} + 2\nabla\log\pi_{T-t}(\overleftarrow{\mathbf{X}}_{t})) \mathrm{d}t + \sqrt{2}\mathrm{d}\mathbf{B}_{t}$$
- Exactly the same marginal $\overleftarrow{\pi}_{T} = \pi_{0} = p^{\star}$
- Suppose: 
	- (i). We know $\pi_{T-t}$
	- (ii). $\overleftarrow{\mathbf{X}}_{0} \sim \pi_{T}$
	- (iii). Exact simulation

**Discussion**.
**For (i)**. Goal: find $\mathbf{s} \in \mathcal{F}$ to minimize
$$\begin{align*}
L_{t}(\mathbf{s}) &=  \mathbb{E} [ \Vert \nabla\log\pi_{T-t}(\overleftarrow{\mathbf{X}}_{t}) - \mathbf{s}_{t}(\overleftarrow{\mathbf{X}}_{t}) \Vert^{2} ]\\
&= \mathbb{E}_{\pi_{T-t}}[\Vert \nabla\log\pi_{T-t}(\mathbf{X}) - \mathbf{s}_{t}(\mathbf{X}) \Vert]\\
&= \mathbb{E}_{\pi_{T-t}}[ \Vert \mathbf{s}_{t}(\mathbf{X}) \Vert^{2} + 2\nabla \cdot \mathbf{s}_{t}(\mathbf{X}) ]
\end{align*}$$
Let global loss function $L(\mathbf{s}) = \int_{0}^{T} w(t) L_{t} (\mathbf{s}) \mathrm{d}t$
Training of diffusion generative model:
- Observed data $\{X^{(i)}\}_{i=1}^{n}$
- For each $i$, randomly pick $t_{i} \sim w$
- Simulate $\mathrm{d}\mathbf{X}_{t} = -\mathbf{X}_{t}\mathrm{d}t + \sqrt{2}\mathrm{d}\mathbf{B}_{t}$ from time $0$ to $T-t_{i}$ with $\mathbf{X}_{0}^{(i)} = \mathbf{X}^{(i)}$ (Can be simulated exactly). Then we get $\mathbf{X}_{t_{i}}^{(i)}$
- Solve $\min\limits_{\mathbf{s}\in\mathcal{F}} \frac{1}{n} \sum\limits_{i=1}^{n}\Vert \mathbf{s}(\mathbf{X}_{t_{i}}^{(i)}) \Vert^{2} + 2\nabla \cdot \mathbf{s}(\mathbf{X}_{t_{i}}^{(i)})$. (Can also use multiple time points)
 **Statistical error analysis using aforementioned framework**.
 e.g. For Holder-smooth $p^{\star}$. With careful choice of $\mathcal{F}$, we can achieve minimax (optimal) rate of convergence.

**For (ii)**. We know that $\pi_{T}$ is close enough to $\mathcal{N}(0, \mathbb{I}_{d})$, 
$$d_\mathrm{TV}(\pi_{T}, \mathcal{N}(0, \mathbb{I}_{d})) \lesssim e^{-T}$$
**For (iii)**. We have discussed it.

Alternative ways of investing the process.
$$
\begin{align*}
\partial_{t} \overleftarrow{\pi}_{t} &= -\nabla\cdot(\mathbf{X}\overleftarrow{\pi}_{t}) - \Delta \overleftarrow{\pi}_{t} \\
&= -\nabla\cdot ( \overleftarrow{\pi}_{t} (\mathbf{X}+\nabla\log\overleftarrow{\pi}_{t}))
\end{align*}
$$
Corresponding flow in original space
$$\mathrm{d}\overleftarrow{\mathbf{X}}_{t} = (\overleftarrow{\mathbf{X}}_{t} + \nabla\log\pi_{T-t}(\overleftarrow{\mathbf{X}}_{t}))\mathrm{d}t$$
We only need to simulate ODE. (Only randomly from $\overleftarrow{\mathbf{X}}_{0} \sim \mathcal{N}(0, \mathbb{I}_{d})$)

ODE might not have enough randomness, but can help with faster generation. (Sometimes it's easier to learn solution mapping directly.)

**Examples of extension**.
If we want to sample from $p^{\star}(\mathbf{x}|\mathbf{c})$ (e.g. $\mathbf{c}$ is text description of image)
(a). learn $\nabla\log\pi_{t}(\mathbf{x}|\mathbf{c})$ directly. ($\mathbf{s}_{t}(\mathbf{x};\mathbf{c})$)
(b). Classifier guidance ($w$ is training parameter)
$$\nabla_\mathbf{x} \log \pi_{t} (\mathbf{x}|\mathbf{c}) = \nabla_\mathbf{x} \log \pi_{t} (\mathbf{x}) + {\color{red}{(w)}}\nabla_\mathbf{x} \log p(\mathbf{c|\mathbf{X}_{t} = \mathbf{x}})$$
(c). Classifier-free guidance
$$\nabla_\mathbf{x} \log\pi_{t}(\mathbf{x}) + w(\nabla_\mathbf{x}\log\pi_{t}(\mathbf{x}|\mathbf{c}) - \nabla_\mathbf{x}\log\pi_{t}(\mathbf{x}))$$

**e.g.** from 2D to 3D (Dreamfusion)
Use a pretrained 2D diffusion model.
$$\theta \in \mathbb{R}^{d} \xrightarrow{} \text{3D shape} \xrightarrow{projection} \text{2D image}$$
Then, 
$$\mathop{\mathrm{minimize}}\limits_{\theta} \mathbb{E}[ \Vert \substack{\text{projection of shape} \\ \text{generated by }\theta} - \substack{\text{image from} \\ \text{diffusion model}} \Vert^{2} ]$$

Fine-tuning.
$$\mathrm{d} \overleftarrow{\mathbf{X}}_{t} = (\overleftarrow{\mathbf{X}}_{t} + 2\nabla\log\pi_{T-t}(\overleftarrow{\mathbf{X}}_{t}))\mathrm{d}t + \mathbf{a}_{t}\mathrm{d}t + \sqrt{2}\mathrm{d}\mathbf{B}_{t}$$
where $\mathbf{a}_{t}$ is adopted process.

**Goal**: 
$$\mathop{\mathrm{minimize}} \mathbb{E}[\mathrm{loss}(\overleftarrow{\mathbf{X}}_{T})] + \alpha \mathcal{D}_\mathrm{KL}(\mathbb{P}_{[0,T]}\Vert\mathbb{P}_{[0,T]}^{\star})$$
- Improve images' visual quality. (e.g. reward model learned from human performance.)
- From experimental data in AI for Science.
- (mostly in LLM) Verifiable reward in reasoning.


Recall score matching objective
$$\min_\mathbf{s} \mathbb{E} [ \Vert \nabla\log\pi_{t}({\mathbf{X}}_{t}) - \mathbf{s}_{t}({\mathbf{X}}_{t}) \Vert^{2} ]$$
(a). Implies score matching $\min_\mathbf{s} \mathbb{E}[ \Vert \mathbf{s}_{t}(\mathbf{X}) \Vert^{2} + 2\nabla \cdot \mathbf{s}_{t}(\mathbf{X}) ]$
(b). Sliced score matching
$$\min_\mathbf{s} \mathbb{E} [\Vert \mathbf{s}_{t}(\mathbf{X}_{t})\Vert^{2} + 2 \nu^{\top} \nabla_\mathbf{x} (\nu^{\top} \mathbf{s}_{t}(\mathbf{X}_{t}))]$$
where $\nu \sim \mathcal{N}(0, \mathbb{I}_{d})$
(c). Denoising score matching
$$\min_\mathbf{s} \mathbb{E} [ \Vert \mathbf{s}_{t}(\mathbf{X}_{t}) - \nabla_\mathbf{y} \log p_{t} (\mathbf{X}_{t} | \mathbf{X}_{0}) \Vert^{2} ]$$
(We denote $\nabla_\mathbf{y} p_{t} (\mathbf{y} | \mathbf{x})$ as derivative w.r.t. the first variable)
$$\begin{align*}
\text{DSM objective} &= \mathbb{E} [\Vert \mathbf{s}_{t}(\mathbf{X}_{t})\Vert^{2}] - 2\mathbb{E}[ \langle \mathbf{s}_{t}(\mathbf{X}_{t}), \nabla_\mathbf{y}\log p_{t}(\mathbf{X}_{t}|\mathbf{X}_{0}) \rangle ] + \text{sth indp of }\mathbf{s}\\
\end{align*}$$
The second term is
$$\begin{align*}
\mathbb{E}[ \langle \mathbf{s}_{t}(\mathbf{X}_{t}), \nabla_\mathbf{y}\log p_{t}(\mathbf{X}_{t}|\mathbf{X}_{0}) \rangle] 
&=  \iint \mathbf{s}_{t}(\mathbf{y})^{\top} \nabla_\mathbf{y} \log p_{t}(\mathbf{y}|\mathbf{x}) \pi_{0}(\mathbf{x}) p_{t}(\mathbf{y}|\mathbf{x}) \mathrm{d}\mathbf{x}\mathrm{d}\mathbf{y}\\
&= \int \mathbf{s}_{t}(\mathbf{y})^{\top} \nabla_\mathbf{y} \left(\int \pi_{0}(\mathbf{x}) p_{t}(\mathbf{y}|\mathbf{x}) \mathrm{d}\mathbf{x} \right) \mathrm{d}\mathbf{y}\\
&= \int \mathbf{s}_{t}(\mathbf{y})^{\top} \nabla\pi_{t}(\mathbf{y}) \mathrm{d}\mathbf{y}\\
&= \mathbb{E}[\langle \mathbf{s}_{t}(\mathbf{X}_{t}), \nabla\log\pi_{t}(\mathbf{X}_{t}) \rangle]
\end{align*}$$
Computing DSM objective
$$\mathrm{loss}_{t}(\mathbf{s}_{t}) = \mathbb{E} [ \Vert \mathbf{s}_{t}(\mathbf{X}_{t}) - \nabla_\mathbf{y} \log p_{t} (\mathbf{X}_{t} | \mathbf{X}_{0}) \Vert^{2} ]$$
where $p_{t}(\cdot | \mathbf{X}_{0}) \sim \mathcal{N}(\mu_{t}(\mathbf{X}_{0}), \sigma_{t}\mathbb{I}_{d})$. (For OU process, $\mu_{t}(\mathbf{X}_{0}) = e^{-t}\mathbf{X}_{0}$, $\sigma_{t} = 1-e^{-2t}$)
Then, 
$$\min_\mathbf{s} \mathbb{E} \left[\left\Vert \mathbf{s}_{t}(\mathbf{X}_{t}) + \frac{\mathbf{X}_{t} - \mu_{t}(\mathbf{X}_{0})}{\sigma_{t}^{2}} \right\Vert^{2}\right]$$

# Lecture 7. RL basics, value functions and value learning

## Discrete-Time RL basics

**Notations**.
- $\mathcal{S}$: State space.
- $\mathcal{A}$: Action space.
- $r$: Reward function.
- ${P}$: Transition dynamics.

For MDP (Markov Decision Process) $\cdots \to s_{t-1} \xrightarrow{P(\cdot|s_{t-1}, A_{t-1})} s_{t} \xrightarrow{P(\cdot|s_{t}, A_{t})} s_{t+1} \to \cdots$

**Value functions**.
- Finite horizon, time: $1, \cdots, T$
$$\begin{align*}
v_{t}^{\pi} (x) &=  \mathbb{E}_{\pi} \left[\sum\limits_{s=t}^{T} R_{s} \bigg| X_{t}=x\right] \\
v_{t}^{\star}(x) &= \max_{\pi} v_{t}^{\pi}(x)
\end{align*}$$
- Discounted, time $0, 1, 2, \cdots$. $\gamma \in (0, 1)$ is called discounted factor.
$$\begin{align*}
v^{\pi}(x) &= \mathbb{E}_{\pi} \left[\sum\limits_{t=0}^{+\infty} \gamma^{t} R_{t} \bigg| X_{0} =x\right]\\
v^{\star}(x) &= \max_{\pi} v^{\pi} (x)
\end{align*}$$
- Infinite horizon, undiscounted. "average reward RL" (Often approximated well by discounted problems)
$$\max_{\pi} \lim_{T\to+\infty} \frac{1}{T}\sum\limits_{t=0}^{T-1}\mathbb{E}_{\pi}[R_{t}]$$
The discussion mainly focuses on discounted case for simplicity.

Value function satisfies,
$$
\begin{align*}
v^{\pi}(x) &=  \mathbb{E}_{\pi}[r(x, A) + \gamma v^{\pi}(X^{+})] = J^{\pi}(v^{\pi})\\
v^{\star}(x) &= \max_{a \in \mathcal{A}} \mathbb{E}_{X^{+}\sim p(\cdot|x, a)} [r(x, a) + \gamma v^{\star}(X^{+})] = J^{\star}(v^{\star})
\end{align*}
$$
Define q-function ,
$$
\begin{align*}
Q^{\pi}(x, a) &=  \mathbb{E}_{\pi} \left[\sum\limits_{t=0}^{T} \gamma^{t} R_{t} \mid X_{0}=x, A_{0} = a\right] \\
Q^{\star}(x, a) &= \max_{\pi} Q^{\pi}(x, a)
\end{align*}
$$
Similarly, $Q^{\pi}$, $Q^{\star}$ also satisfy Bellman equations.
$$
\begin{align*}
Q^{\pi}(x, a) &= \mathbb{E}_{\pi}[r(x, a) + \gamma Q^{\pi}(X^{+}, A^{+})] = J_{Q}^{\pi}(Q^{\pi})\\
Q^{\star}(x, a) &= \mathbb{E}_{\pi} [r(x, a)] + \gamma \mathbb{E}\left[\max_{a^{+}\in\mathcal{A}}Q^{\star}(x^{+}, a^{+})\right]  = J_{Q}^{\star}(Q^{\star})
\end{align*}
$$
**Fact**. $J^{\pi}$, $J^{\star}$, $J_{Q}^{\pi}$, and $J_{Q}^{\star}$ are $\gamma$-contractions under $\Vert\cdot\Vert_{\infty}$ norm. (i.e.$\Vert Jv_{1} - Jv_{2}\Vert_{\infty}\le\gamma\Vert v_{1}-v_{2}\Vert_\infty$)

Fixed-point algorithms: Value Iteration and Policy Iteration.

**Value Iteration**: $Q^{(t+1)}=J^{\star}_{Q}(Q^{(t)})$ ($t=0, 1, 2, \cdots$) Until $\Vert Q^{(t)} - Q^{\star}\Vert_{\infty} \lt \gamma^{t} \Vert Q^{(0)}-Q^{\star}\Vert_{\infty}$

**Policy Iteration**: $Q^{(t+1)} = (\mathbb{I}-\gamma\mathbf{P}_{\pi}(t))^{-1} r_{\pi^{(t)}}$, $\pi^{(t+1)}= \mathrm{greedy-optimal}(Q^{(t+1)})$ Until $\Vert Q^{(t)} - Q^{\star}\Vert_{\infty} \lt \gamma^{t} \Vert Q^{(0)}-Q^{\star}\Vert_{\infty}$ (The convergence can actually be faster like a Newton method)

## Data-driven solutions: RL

Naive idea: plug-in solution.
**e.g.** for value iteration.
$$\begin{align*}
Q^{(t+1)} &= J_{Q}^{\star} (Q^{(t)})\\
\hat{Q}_{n}^{(t+1)} &= \hat{J}_{n} (\hat{Q}_{n}^{(t)})
\end{align*}$$
where $\hat{J}_n$ is empirical average estimation for Bellman equation

Alternatively, stochastic approximation:
$$\hat{Q}^{(t+1)} = (1-\beta_{t})\hat{Q}^{(t)} + \beta_{t} J_{t+1}(\hat{Q}^{(t)})$$
It's Q-learning algorithm

**From fixed points to projected fixed points**

Function approximation: Use function class $\mathcal{F}$ to approximate $Q^{\star}$
$$\bar{Q} = \Pi_\mathcal{F}\circ J^{\star}(\bar{Q})$$
Ideally, we want to find $\Pi_\mathcal{F} (Q^{\star})$

**Q-learning with function approximation**
$$\hat{Q}_{n} = \hat{\Pi}_{n} \circ \hat{J}_{n} (\hat{Q}_{n})$$
In practice, update the operator each time using new data point.

Similarly, for policy iteration: "**Actor-Critic algorithm**"

Questions:
- Easy to solve projected fixed point?
- Does projected fixed point always lead to good solution?

In general negative, with some known hardness results.
- High-level reason: $L^{2}$-$L^{\infty}$ mismatch.
- Some special cases are solvable
	- For policy evaluation $\Vert J^{\pi}v_{1} - J^{\pi}v_{2}\Vert_{L^{2}} \le \gamma\Vert v_{1} - v_{2}\Vert_{L^{2}}$
$$\mathbb{E}[\Vert \hat{v}_{n}^{\pi} - v^{\pi}\Vert_{L^{2}}^{2}] \le \frac{1}{1-\gamma} \inf_{v \in \mathcal{F}} \Vert v - v^{\pi}\Vert_{L^{2}}^{2} + \epsilon_{\pi}^{2}$$
	- For optimal Q-function
**e.g.** Optimal stopping. $L^{2}$ contraction
**e.g.** Bellman closure / linear MDP

# Lecture 8. Continuous-time control problem, HJB equations, continuous-time RL

## Continuous-Time RL

For diffusion process with control: 
$$\mathrm{d}\mathbf{X}_{t} = \mathbf{b}_{t}(\mathbf{X}_{t}, A_{t})\mathrm{d}t + \Sigma^{\frac{1}{2}}_{t}(\mathbf{X}_{t}, A_{t})\mathrm{d}\mathbf{B}_{t}$$
Action $A_{t} = \pi_{t}(\mathbf{X}_{t})$
Define value function:
$$v^{\pi}(\mathbf{x}) = 
\begin{cases}
\mathbb{E}_{\pi} \left[ \int_{t}^{T} r(\mathbf{X}_{s}) \mathrm{d}s | \mathbf{X}_{t} = \mathbf{x}\right]  & \text{, finite-horizon}\\ 
\mathbb{E}_{\pi}\left[\int_{0}^{+\infty}e^{-\beta t} r(\mathbf{X}_{t})\mathrm{d}t | \mathbf{X}_{0} = \mathbf{x}\right] & \text{, discounted}
\end{cases}$$
$$v^{\star}(x) = \max_{\pi} v^{\pi}(x)$$
Observation/action application every $\Delta t$ time
- This is just a discrete-time MDP
- We can run any algorithm

**Underlying differential equation**.

Consider finite-horizon, controlled Markov diffusion, for $[t, t+\Delta t]$
$$
v^{\star}_{t}(\mathbf{x}) \approx \max_{a\in\mathcal{A}} \mathbb{E}\left[ \underbrace{\int_{0}^{\Delta t} r_{t+s}(\mathbf{X}_{t+s}, a)\mathrm{d}s}_{\approx r_{t+s}(\mathbf{X}_{t+s}, a) \Delta t} + v_{t+\Delta t}^{\star} (\mathbf{X}_{t+\Delta t}) \bigg| \mathbf{X}_{t} = \mathbf{x} \right]
$$
By Ito's formulae,
$$
\begin{align*}
v_{t+\Delta t}^{\star}(\mathbf{X}_{t+\Delta t}) &= v_{t}^{\star}(\mathbf{x})
+ \int_{0}^{\Delta t} \frac{\partial v_{t+s}}{\partial s}(\mathbf{X}_{t+s})\mathrm{d}s + \int_{0}^{\Delta t}\langle
\nabla v_{t+s}^{\star}(\mathbf{X}_{t+s}), \mathbf{b}_{t+s}(\mathbf{X}_{t+s})
\rangle \mathrm{d}s \\
&+ \underbrace{\int_{0}^{\Delta t} \nabla v_{t+s}^{\star}(\mathbf{X}_{t+s})^{\top} \Sigma^{\frac{1}{2}}_{t+s}(\mathbf{X}_{t+s})\mathrm{d}\mathbf{B}_{s}}_{\text{martingale}}
 + \frac{1}{2} \int_{0}^{\Delta t} \mathrm{Tr}(\Sigma_{t+s}(\mathbf{X}_{t+s}) \nabla^{2} v_{t+s}^{\star}(\mathbf{X}_{t+s})) \mathrm{d}s
\end{align*}
$$
Then, we get HJB (Hamilton-Jacobi-Bellman) equation:
$$\color{red}{\boxed{\partial_{t} v_{t}^{\star}(\mathbf{x}) + \max_{a\in\mathcal{A}} r_{t}(\mathbf{x}, a) + \langle \nabla v_{t}^{\star}(\mathbf{x}), \mathbf{b}_{t}(\mathbf{x}, a) \rangle +
\frac{1}{2} \mathrm{Tr} (\nabla^{2}v_{t}^{\star}(\mathbf{x})\Sigma_{t}(\mathbf{x}, a)) = 0}}$$

Similarly, for discounted case, we have
$$\color{red}{\boxed{\beta v^{\star}(\mathbf{x}) = \max_{a\in\mathcal{A}} r(\mathbf{x}, a) + \langle \nabla v^{\star}(\mathbf{x}), \mathbf{b}(\mathbf{x}, a) \rangle +
\frac{1}{2} \mathrm{Tr} (\nabla^{2}v^{\star}(\mathbf{x})\Sigma(\mathbf{x}, a)) = 0}}$$
For policy evaluation,
$$
\color{red}{\boxed{
\begin{cases}
\partial_{t} v_{t}^{\pi}(\mathbf{x}) + r_{t}(\mathbf{x}) + \mathcal{L}_{\pi} v^{\pi}_{t}(\mathbf{x}) = 0  & \text{, finite-horizon}\\ 
\beta v^{\pi}(\mathbf{x}) = r^{\pi}(\mathbf{x}) + \mathcal{L}_{\pi}v^{\pi}(\mathbf{x})  & \text{, discounted}
\end{cases}
}}
$$
where $\mathcal{L}_{\pi}$ is the diffusion generator

For discounted policy evaluation.
Discrete RL guarantee gives,
$$
\mathbb{E}[\Vert \hat{v} - v^{\pi}\Vert_{L^{2}}^{2}] \le \underbrace{\frac{1}{1-\gamma}}_{\approx \frac{1}{\beta\Delta t}} \inf_{v \in \mathcal{F}} \Vert v - v^{\pi}\Vert_{L^{2}}^{2} + \epsilon_{\text{statistics}}^{2}
$$

# Lecture 9. Advanced topics in continuous-time RL, applications to diffusion model fine-tuning

**Recall**. policy evaluation
$$\begin{align*}
\mathrm{d}\mathbf{X}_{t}^{\pi} &=  \mathbf{b}^{\pi} (\mathbf{X}_{t}^{\pi}) \mathrm{d}t + \Sigma^{\pi}(\mathbf{X}_{t}^{\pi})\mathrm{d}\mathbf{B}_{t}\\
v^{\pi}(\mathbf{x}) &= \int_{0}^{+\infty}\exp(-\beta t) \mathbb{E}_{\pi} [r(\mathbf{X}_{t})|\mathbf{X}_{0} = \mathbf{x}] \mathrm{d}t
\end{align*}$$
Observe $\mathbf{X}_{0}$, $\mathbf{X}_{\Delta t}$, $\mathbf{X}_{2\Delta t}$, $\cdots$, where $\mathbf{X}_{0} \sim \mathrm{stationary} (\text{denoted as }\mu)$

**Projected fixed point**
$$\bar{v} = \Pi_{\mathcal{F}, L^{2}(\mu)}\circ J(\bar{v}) $$
Solvable using data (e.g. $\mathcal{F} = \mathbb{S} = \mathrm{span}(\mathbf{\Phi}_{1}, \cdots, \mathbf{\Phi}_{m})$ is linear subspace)

Its empirical version is,
$$\frac{1}{n} \sum\limits_{i=1}^{n} \mathbf{\Phi}(\mathbf{X}_{i\Delta t}) (\mathbf{\Phi}(\mathbf{X}_{i\Delta t}) - \exp(-\beta\Delta t)\mathbf{\Phi}(\mathbf{X}_{(i+1)\Delta t}))^{\top} \hat{\theta} = \frac{1}{n} \sum\limits_{i=1}^{n}r_{i\Delta t} \mathbf{\Phi}(\mathbf{X}_{i\Delta t}) \Delta t$$
Then,
$$\Vert \bar{v} - v^{\pi} \Vert_{L^{2}(\mu)} \le \underbrace{\frac{1}{\sqrt{1-e^{-\beta \Delta t}}}}_{O\left(\frac{1}{\sqrt{\Delta t}}\right)} \inf_{v \in \mathcal{S}} \Vert \bar{v} - v^{\pi} \Vert_{L^{2}(\mu)} {\color{red}{\left(+\sqrt{\Delta t}\right)}}$$
On the other hand, $v^{\pi}$ satisfies $\beta v^{\pi} - \mathcal{L}v^{\pi} = r$, where $\mathcal{L}$ is diffusion generator.
If coefficients were known, Galerkin method guarantees good approximations.

Galerkin method: find function $\bar{v}\in\mathbb{S}$, s.t.
$$\langle (\beta-\mathcal{L})\bar{v}-r, f \rangle_{L^{2}(\mu)}=0, \forall f\in\mathbb{S}$$
How well does Galerkin method work?
$$\langle (\beta-\mathcal{L})\bar{v}-(\beta-\mathcal{L})v^{\pi}, f \rangle_{L^{2}(\mu)}=0, \forall f\in\mathbb{S}$$
Let $\tilde{v} = \mathop{\arg\min}_{v\in\mathbb{S}}\lVert v^{\pi}-v\rVert$, $f=\bar{v}-\tilde{v}$, then,
$$\langle(\beta-\mathcal{L})(\bar{v}-v^{\pi}),\bar{v}-\tilde{v}\rangle=0$$

$\color{red}{\text{Hope:}}$
$$\begin{align*}
{\color{red}{\lambda\lVert \bar{v}-\tilde{v}\rVert_{\color{blue}{H^{1}}}}}&{\color{red}{\le}} \langle(\beta-\mathcal{L})(\bar{v}-\tilde{v}),\bar{v}-\tilde{v}\rangle_{L^{2}(\mu)}\\
&= \langle(\beta-\mathcal{L})(\tilde{v}-v^{\pi}),\bar{v}-\tilde{v}\rangle_{L^{2}(\mu)}\\
&{\color{red}{\le}} {\color{red}{L\lVert \tilde{v}-\bar{v}\rVert}}_{\color{blue}{H^{1}}}{\color{red}{\lVert \tilde{v}-v^{\pi}\rVert}}_{\color{blue}{H^{1}}}
\end{align*}
$$
where,
$$\lVert f\rVert_{H^{1}(\mu)}=\mathbb{E}_{\mu}|f(X)|^{2} + \mathbb{E}_{\mu}|\nabla f(X)|^{2}$$

Lax-Milgram theorem: $\exists !$ PDE solution under these estimates.
Proof. 
Lower bound:
$$\begin{align*}
\langle(\beta-\mathcal{L})f,f\rangle_{L^{2}(\mu)}
&= \int\left(\beta f(x)-b^{\top}\nabla f(x) - \frac{1}{2}\mathrm{Tr}(\Sigma\cdot\nabla^{2}f)(x)\right)f(x)\mu(x)\mathrm{d}x\\
&= \beta\lVert f\rVert_{L^{2}(\mu)} + \frac{1}{2}\int\nabla f(x)^{\top} \nabla(\Sigma f\mu)\mathrm{d}x+\int f(x)\nabla(f b \mu)\mathrm{d}x\\
&\quad{\color{red}{\left(\text{using the fact that }-\nabla(b\mu)+ \frac{1}{2}\nabla^{2}(\Sigma\mu)=0\right)}}
\\
&= \beta\lVert f\rVert_{L^{2}(\mu)}+ \frac{1}{2}\int\nabla f(x)^{\top}\Sigma(x)\nabla f(x)\mu(x)\mathrm{d}x
\end{align*}$$
Assuming $\Sigma(x)\succeq\lambda_{\mathrm{min}}\mathbb{I}_{d}$, (uniform ellipticity) for $\lambda_\mathrm{min}\gt 0$, $\forall x\in \mathbb{R}^{d}$
$$\langle (\beta-\mathcal{L})f, f \rangle_{L^{2}(\mu)}\ge \min\left(\beta ,\frac{\lambda_\mathrm{min}}{2}\right)\lVert f \rVert^{2}_{H^{1}}$$
Similarly, 
$$|\langle(\beta-\mathcal{L})f, g\rangle_{L^{2}(\mu)}| \le L \lVert f \rVert^{2}_{H^{1}} \lVert g \rVert^{2}_{H^{1}}$$
So we get (Cea, 1964)
$$
\lVert \bar{v}-v^{\pi}\rVert_{H^{1}(\mu)}\le \frac{L}{\min\left(\beta ,\frac{\lambda_\mathrm{min}}{2}\right)}\inf_{v\in \mathbb{S}}\lVert v-v^{\pi}\rVert_{H^{1}(\mu)}
$$
(Possible extension to "hypo-ellipticity", e.g. underdamped Langevin)

For RL (temporal difference) method,
$$
\bar{v}=\Pi_{\mathbb{S},L^{2}(\mu)}(\exp(-\beta\Delta t)\mathcal{P}_{\Delta t}\bar{v}+r_{\Delta t})
$$
i.e.
$$
\left\langle \underbrace{\frac{(\mathbb{I}-\exp(-\beta\Delta t)\mathcal{P}_{\Delta t})\bar{v}}{\Delta t}}_{\lim_{\Delta t\to0} (\cdots)=\beta-\mathcal{L}} - \underbrace{\frac{r_{\Delta t}}{\Delta t}}_{\approx r} , f\right\rangle_{L^{2}(\mu)} =0, f\in\mathbb{S}
$$
Ideally, we want upper & lower bounds for 
$$\frac{\mathbb{I}-\exp(-\beta\Delta t)\mathcal{P}_{\Delta t}}{\Delta t}$$
for finite $\Delta t$.
i.e. 
$$\left\langle\frac{\mathbb{I}-\exp(-\beta\Delta t)\mathcal{P}_{\Delta t}}{\Delta t} f, f\right\rangle_{L^{2}(\mu)} {\color{red}{\underbrace{\gtrsim}_{?}}} \lVert f\rVert^{2}_{H^{1}(\mu)}$$
- impossible in general
- but under some regularity conditions on basis functions of $\mathbb{S}$, true for $\Delta t\le \Delta t_{\mathrm{thres}}(\mathbb{S})$, the lower bound holds for $\forall f \in \mathbb{S}$
- upper bounds still hold for finite $\Delta t$

Conclusion:
$$\lVert \bar{v}-v^{\pi}\rVert_{H^{1}}\le C\cdot\inf_{v\in\mathbb{S}}\lVert {v}-v^{\pi}\rVert_{H^{1}}$$
$\bar{v}$ is target of TD algo, its statistic error is:
$$
\lVert \bar{v}-\hat{v}_T\rVert_{H^{1}(\mu)}\lesssim\frac{m \cdot t_\mathrm{mix}}{T}\inf_{v\in\mathbb{S}}\lVert {v}-v^{\pi}\rVert_{W^{1,p}}+\underbrace{\frac{g(m)}{T}}_{g(m)=o(m)}$$
- Nonstandard trade-off, policy evaluation is easier than regression.
- Extension to general Q-learning: Possible under certain cases. (ongoing work)

Diffusion fine-tuning:
$$
\min \mathbb{E}\left[
  C(T) + \int_{0}^{T} y(X_{t})\mathrm{d}t
\right]
+ \alpha \mathcal{D}_\mathrm{KL}(\mathbb{P}_{[0,T]}\| \mathbb{P}_{[0,T]}^{\mathrm{pretrained}})
$$
s.t. $\mathrm{d}\mathbf{X}_{t}=\mathbf{b}(\mathbf{X}_{t})\mathrm{d}t + \mathbf{A}_{t}\mathrm{d}t + \sqrt{2}\mathrm{d}\mathbf{B}_{t}$

HJB eq:
$$
\frac{\partial v_{t}^{\star}(x)}{\partial t} + \min_{a\in\mathbb{R}^{d}} [\langle b(x)+a,\nabla v_{t}^{\star}(x)\rangle + \Delta v_{t}^{\star}(x) + y(x)+\alpha|a|^2]=0
$$
then,
$$
\partial_{t} v_{t}^{\star}(x) + \langle b(x),\nabla v_{t}^{\star}(x)\rangle + \Delta v_{t}^{\star}(x) - \frac{1}{4\alpha}|\nabla v_{t}^{\star}(x)|^{2} + y_{t}(x)=0
$$
Exponential transformation, let $f_{t}^{\star}=\exp\left(\frac{{v_{t}^{\star}}}{\alpha}\right)$, then,
$$(\partial_{t}+\alpha y_{t}+\mathcal{L}_{t})f_{t}^{\star}=0$$
boundary condition: $f_{T}^{\star}=\exp( \frac{C}{\alpha} )$

RL-fine-tuning is as easy as regression for diffusion models (and even easier)


## Talk is cheap, show me the code/implementations
Some key points to bridge the gap between theoretical analysis and empirical implementation.

### Denoising diffusion probabilistic model
We have data points $x_{i} \sim p_{\text{data}}$. Given a (invertible) mapping that add noise to images $f: x_{t-1}\mapsto x_{t}$ such that $\lim_{T\to\infty}f^{T}(x_{0})\approx z\sim \mathcal{N}(0, \mathbf{I})$, if we have learned $\mu\approx f^{-1}$, we can generate any image distributed in $p_{\text{data}}$ by $\hat{x}=\mu^{T}(z)$, where $z\sim\mathcal{N}(0,\mathbf{I})$.

In DDPM, we formulate $f: x_{t-1}\mapsto x_{t}=\alpha_{t}x_{t-1}+\beta_{t}\epsilon_{t}$, $\epsilon_{t}\sim\mathcal{N}(0,\mathbf{I})$. (Markov? Maybe helpful)
Then we have $x_{T}=(\prod_{i=1}^{T}\alpha_{i})x_{0} + \sum_{i=1}^{T}(\prod_{j=i+1}^{T}\alpha_{j})\beta_{i}\epsilon_{i}$. If we have $\alpha_{i}^{2}+\beta_{i}^{2}=1$, then the noisy term $\sum_{i=1}^{T}(\prod_{j=i+1}^{T}\alpha_{j})\beta_{i}\epsilon_{i}\sim\mathcal{N}(0,1-(\prod_{i=1}^{T}\alpha_{i})^{2})$. Denote $\bar{\alpha}_{T} = \prod_{i=1}^{T}\alpha_{i}$, $\bar{\beta}_{T}=1-\bar{\alpha}_{T}^{2}$, then we can add noise by one step sampling:
$$
x_{T}=\bar{\alpha}_{T}x_{0} + \bar{\beta}_{T}\bar{\epsilon}_{T},\quad \bar{\epsilon}_{T}\sim\mathcal{N}(0,\mathbf{I})
$$

**How to train DDPM empirically?**
We have training dataset $\mathcal{D}=\{(x_{t-1}, x_{t})\}$ consisting of data pairs.
Simple idea: Since we get $x_{t}$ by adding noise $x_{t}=\alpha_{t}x_{t-1}+\beta_{t}\epsilon_{t}$, we can denoise it by $x_{t-1}= \frac{1}{\alpha_{t}} (x_{t}-\beta_{t}\epsilon_{t})$. Then we can:
- Learn a model $\mu_{\theta}: x_{t}\mapsto x_{t-1}$, then $\theta=\mathop{\arg\min}_{\theta}\mathcal{L}=\mathop{\arg\min}_{\theta}\lVert x_{t-1}-\mu_{\theta}(x_{t})\rVert_{2}^{2}$
- Formulate $\mu_{\theta}(x_{t})=\frac{1}{\alpha_{t}} (x_{t}-\beta_{t}\epsilon_{\theta}(x_{t}, t))$
- Then we have the new objective $\mathcal{L}=\lVert x_{t-1}-\mu_{\theta}(x_{t})\rVert_{2}^{2}= (\frac{\beta_{t}}{\alpha_{t}})^{2} \lVert \epsilon_{t}-\epsilon_{\theta}(x_{t},t)\rVert_{2}^{2}$
We can sample $x_{t}=\bar{\alpha}_{t}x_{0} + \bar{\beta}_{t}\bar{\epsilon}_{t}$. However, note that $\epsilon_{t}$ and $\bar{\epsilon}_{t}$ are not independent, so we sample $x_{t}=\alpha_{t}x_{t-1}+\beta_{t}\epsilon_{t}=\alpha_{t}(\bar{\alpha}_{t-1}x_{0}+\bar{\beta}_{t-1}\bar{\epsilon}_{t-1})+\beta_{t}\epsilon_{t}$
Then the objective is (remove weight term): 
$$\begin{align*}
\mathcal{L} &= \lVert \epsilon_{t}-\epsilon_{\theta}(x_{t},t)\rVert_{2}^{2}\\
&= \lVert \epsilon_{t}-\epsilon_{\theta}(\bar{\alpha}_{t}x_{0}+\alpha_{t}\bar{\beta}_{t-1}\bar{\epsilon}_{t-1}+\beta_{t}\epsilon_{t},t)\rVert_{2}^{2}
\end{align*}
$$
During training phase, we should sample $x_{0}\sim p_{\text{data}}$, $t\sim \mathcal{U}(1,T)$ and $\epsilon_{t},\bar{\epsilon}_{t-1}\overset{\text{i.i.d.}}{\sim}\mathcal{N}(0,\mathbf{I})$ and minimize $\mathbb{E}(\mathcal{L})$. But sampling 4 r.v. makes $\text{Var}(\mathcal{L})$ too big.
Note that $(\alpha_{t}\bar{\beta}_{t-1})^{2}+\beta_{t}^{2}=\bar{\beta}_{t}^{2}$, so we have,
$$\begin{align*}
\alpha_{t}\bar{\beta}_{t-1}\bar{\epsilon}_{t-1} + \beta_{t}\epsilon_{t} &\sim \bar{\beta}_{t} \epsilon\\
\beta_{t}\bar{\epsilon}_{t-1} - \alpha_{t}\bar{\beta}_{t-1}\epsilon_{t} &\sim \bar{\beta}_{t} \omega
\end{align*}$$
where $\epsilon, \omega \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,\mathbf{I})$. Then,
$$\epsilon\omega^{\top}=\frac{1}{\bar{\beta}_{t}^{2}}[ 
\alpha_{t}\bar{\beta}_{t-1}\beta_{t}(\bar{\epsilon}_{t-1}\bar{\epsilon}_{t-1}^{\top}-\epsilon_{t}\epsilon_{t}^{\top}) + (\beta_{t}^{2} - \alpha_{t}^{2}\bar{\beta}_{t-1}^{2})\bar{\epsilon}_{t-1} \epsilon_{t}^{\top}
]$$
So we have $\mathbb{E}(\epsilon\omega^{\top})=0$, i.e. $\epsilon\perp\omega$. The final objective is
$$\begin{align*}
\mathbb{E}(\mathcal{L}) 
&=  \mathbb{E}_{\bar{\epsilon}_{t-1}, \epsilon_{t}\overset{\text{i.i.d.}}{\sim}\mathcal{N}(0,\mathbf{I})} [\lVert \epsilon_{t}-\epsilon_{\theta}(\bar{\alpha}_{t}x_{0}+\alpha_{t}\bar{\beta}_{t-1}\bar{\epsilon}_{t-1}+\beta_{t}\epsilon_{t},t)\rVert_{2}^{2}]\\
&= \mathbb{E}_{\epsilon,\omega\overset{\text{i.i.d.}}{\sim}\mathcal{N}(0,\mathbf{I})} \left[\left\lVert \frac{1}{\bar{\beta}_{t}} (\beta_{t}\epsilon-\alpha_{t}\bar{\beta}_{t-1}\omega)-\epsilon_{\theta}(
\bar{\alpha}_{t}x_{0}+\bar{\beta}_{t}\epsilon
,t)\right\rVert_{2}^{2}\right]\\
&= \frac{\beta_{t}^{2}}{\bar{\beta}_{t}^{2}} \mathbb{E}_{\epsilon\sim\mathcal{N}(0,\mathbf{I})} \left[\left\lVert
\epsilon- \frac{\bar{\beta}_{t}^{2}}{\beta_{t}^{2}}\epsilon_{\theta}(\bar{\alpha}_{t}x_{0}+\bar{\beta}_{t}\epsilon,t) 
\right\rVert_{2}^{2}\right]
\end{align*}
$$
And the empirical loss is
$$
\mathcal{L}_{\text{empirical}} = \frac{1}{n}
\sum\limits\left\lVert
\epsilon - \frac{\bar{\beta}_{t}^{2}}{\beta_{t}^{2}} \epsilon_{\theta}(
\bar{\alpha}_{t}x_{0}+\bar{\beta}_{t}\epsilon,t
)\right\rVert_{2}^{2}
$$
where $x_{0}\sim p_{\text{data}},\epsilon\sim\mathcal{N}(0,\mathbf{I}),t\sim\mathcal{U}(1,T)$.