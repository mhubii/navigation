#include "keyboard.h"

#include <linux/input.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>

// Constructor.
Keyboard::Keyboard() {

	found_device_ = -1;
	memset(key_map_, 0, sizeof(key_map_));
}

// Destructor.
Keyboard::~Keyboard() {

}

// Create keyboard device.
Keyboard* Keyboard::Create(const char* path) {

	if (!path) {
		
		return NULL;
	}

	const int fd = open(path, O_RDONLY);

	if (fd == -1) {
	
		printf("Keyboard -- failed to open %s: %s\n", path, strerror(errno));
	}

	Keyboard* kb = new Keyboard();

	kb->found_device_ = fd;
	kb->path_ = path;

	return kb;
}

// Poll.
bool Keyboard::Poll(uint32_t timeout) {

	const uint32_t max_ev = 64;
	struct input_event ev[max_ev];

	fd_set fds;
	FD_ZERO(&fds);
	FD_SET(found_device_, &fds);

	struct timeval tv;

	tv.tv_sec = 0;
	tv.tv_usec = timeout*1000;

	const int result = select(found_device_ + 1, &fds, NULL, NULL, &tv);

	if (result == -1) {
	
		printf("Keyboard -- select() timed out...\n");
	}
	else if (result == 0) {

		return false;
	}

	const int bytes_read = read(found_device_, ev, sizeof(struct input_event)*max_ev);

	if (bytes_read < (int)sizeof(struct input_event)) {

		printf("Keyboard -- read() expected %d bytes, got %d\n", (int)sizeof(struct input_event), bytes_read);

		return false;
	}

	const int num_ev = bytes_read/sizeof(struct input_event);

	for (int i = 0; i < num_ev; i++) {

		if (ev[i].type != EV_KEY) {
			
			continue;
		}
		if (ev[i].value < 0 || ev[i].value > 2) {
	
			continue;
		}
		if (ev[i].code >= MAX_KEYS) {
	
			continue;
		}
		
		key_map_[ev[i].code] = (ev[i].value == 0) ? false : true;
	}
	
	return true;
}

// KeyDown.
bool Keyboard::KeyDown(uint32_t code) const {

	if (code >= MAX_KEYS) {
	
		return false;
	}
	
	return key_map_[code];
}
