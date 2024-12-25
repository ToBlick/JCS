To run on greene GPU:
- Make sure to install jax with cuda backend: pip install -U "jax[cuda12]"
- Double check in a notebook/script that the correct backend is used with 
    from jax.lib import xla_bridge
    print(xla_bridge.get_backend().platform)