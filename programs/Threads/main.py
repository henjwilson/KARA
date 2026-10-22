import threading
import time
from controller_def import init_controller, Controller_main
from opencv_def import init_opencv, opencv_main

# Global system flags (similar to FreeRTOS event groups/status flags)
system_shutdown = threading.Event()

def main():
    # 1. Create the shared components
    controller_wakeup = threading.Event()
    controller_to_opencv = threading.Event()
    opencv_done = threading.Event()
    controller_stop = threading.Event()

    map_lock = threading.Lock()
    active_blocks = {}
    lookup_index = -1

    # 2. Inject everything directly into the controller's scope
    init_controller(
        controller_wakeup,
        controller_to_opencv,
        opencv_done,
        controller_stop,
        lookup_index,
        map_lock,
        active_blocks
    )
    # 3. Spawn and start the worker thread (clean and error-free!)
    controller_task = threading.Thread(target=Controller_main, daemon=True)
    controller_task.start()

    time.sleep(0.5) # Let the thread boot up

    # 4. start up Opencv
    opencv_stop = threading.Event()
    init_opencv(
        controller_to_opencv,
        opencv_done,
        opencv_stop,
        lookup_index,
        map_lock,
        active_blocks
    )
    # 3. Spawn and start the worker thread (clean and error-free!)
    opencv_task = threading.Thread(target=opencv_main, daemon=True)
    opencv_task.start()

    time.sleep(1) # Let the thread boot up

    print("\n--- System Ready ---")
    print("Enter '1' to wake up the controller. Enter '0' to stop and exit.")

    while True:
        # 🛑 THE MAIN THREAD BLOCKS HERE: It freezes and waits for a keyboard event.
        # Your worker threads (Controller / OpenCV) will continue running in the background.
        user_input = None
        if not controller_wakeup.is_set():
            user_input = input("Enter command (1=Run, 0=Exit): ").strip()

        # 🔓 THE MAIN THREAD UNBLOCKS HERE: Execution resumes ONLY after Enter is pressed.
        if user_input == "1":
            print("[Main] Waking up the controller task...")
            controller_wakeup.set()

        elif user_input == "0":
            print("[Main] Shutdown command received. Cleaning up...")
            break

     # 3. Safe Task Cleanup Code
    # Signal the controller that it's time to permanently stop
    controller_stop.set()
    opencv_stop.set()

    # In case the controller is currently stuck sleeping on .wait(),
    # kicking the wakeup event forces it to unblock, see the stop signal, and exit.
    controller_wakeup.set()
    controller_to_opencv.set()


    # This blocks the main execution flow completely until system_shutdown is set
    # system_shutdown.wait()
    controller_task.join()
    opencv_task.join()

    with map_lock:
        active_blocks.clear()
        print("Shared map has been safely cleared.")

    print("[System] All tasks dead. Hardware shutdown complete.")
    print("[Main] Done. Exiting program.")

if __name__ == "__main__":
    main()