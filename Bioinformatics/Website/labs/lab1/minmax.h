static inline int min(int x, int y)
{
	return (x < y ? x : y);
}

static inline int min3(int x, int y, int z)
{
	return (x < y ? (x < z ? x : z) : y);
}

static inline int max(int x, int y)
{
	return (x > y ? x : y);
}

static inline int max3(int x, int y, int z)
{
	return (x > y ? (x > z ? x : z) : y);
}
