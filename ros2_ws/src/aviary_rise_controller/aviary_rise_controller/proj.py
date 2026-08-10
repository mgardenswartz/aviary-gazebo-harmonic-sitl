import jax
import jax.numpy as jnp

@jax.jit
def discrete_projection(
    theta_hat: jax.Array,
    theta_dot_unprojected: jax.Array,
    dt: float,
    theta_bar: float,
    gamma_diag: jax.Array
) -> jax.Array:
    theta_temp = theta_hat + dt * theta_dot_unprojected
    is_inside = jnp.sum(theta_temp**2) <= theta_bar**2
    
    def apply_projection(_: None) -> jax.Array:
        gamma_min = jnp.min(gamma_diag)
        norm_temp = jnp.linalg.norm(theta_temp)
        eta_upper_init = (norm_temp / theta_bar - 1.0) / gamma_min
        init_state = (0.0, eta_upper_init)
        
        def bisection_step(i: int, state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            eta_low, eta_high = state
            eta_mid = 0.5 * (eta_low + eta_high)
            theta_test = theta_temp / (1.0 + eta_mid * gamma_diag)
            val = jnp.sum(theta_test**2) - theta_bar**2
            new_low = jnp.where(val > 0, eta_mid, eta_low)
            new_high = jnp.where(val > 0, eta_high, eta_mid)
            return (new_low, new_high)
        
        final_low, final_high = jax.lax.fori_loop(0, 30, bisection_step, init_state)
        eta_opt = 0.5 * (final_low + final_high)
        return theta_temp / (1.0 + eta_opt * gamma_diag) # type: ignore

    def bypass_projection(_: None) -> jax.Array:
        return theta_temp

    return jax.lax.cond(is_inside, bypass_projection, apply_projection, None) # type: ignore
