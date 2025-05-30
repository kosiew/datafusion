-- Test simple array_map usage
SELECT array_map([1, 4, 9, 16], 'sqrt') as result;
